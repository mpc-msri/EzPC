#include <sytorch/tensor.h>
#include <fstream>
#include <filesystem>
#include <map>

template <typename T>
class SytorchModule {
public:

    Tensor4D<T> activation;
    Backend<T> *backend = new ClearText<T>;
    LayerGraphNode<T> *root = nullptr;
    bool debug = true;
    u64 scale;
    std::map<std::string, LayerGraphNode<T> *> addLayerMap;
    std::map<std::string, LayerGraphNode<T> *> concatLayerMap;
    std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder;

public:

    virtual Tensor4D<T>& _forward(Tensor4D<T> &input) = 0;

    SytorchModule() : activation(0, 0, 0, 0), allNodesInExecutionOrder(0)
    {

    }

    void generateAddLayerMap()
    {
        addLayerMap.clear();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (node->layer->name == "Add") {
                std::string id = "";
                for(auto& parent: node->parents) {
                    id += "|" + std::to_string((uint64_t)(parent));
                }
                addLayerMap[id] = node;
            }
        });
    }

    void generateConcatLayerMap()
    {
        concatLayerMap.clear();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (node->layer->name == "Concat") {
                std::string id = "";
                for(auto& parent: node->parents) {
                    id += "|" + std::to_string((uint64_t)(parent));
                }
                concatLayerMap[id] = node;
            }
        });
    }

    LayerGraphNode<T> *getAddNode(std::vector<Tensor4D<T> *> ips)
    {
        std::string id = "";
        for(auto& ip: ips) {
            id += "|" + std::to_string((uint64_t)(ip->graphNode));
        }
        if (addLayerMap.find(id) == addLayerMap.end()) {
            std::cerr << "Add layer not found" << std::endl;
            exit(1);
        }
        return addLayerMap[id];
    }

    LayerGraphNode<T> *getConcatNode(std::vector<Tensor4D<T> *> ips)
    {
        std::string id = "";
        for(auto& ip: ips) {
            id += "|" + std::to_string((uint64_t)(ip->graphNode));
        }
        if (concatLayerMap.find(id) == concatLayerMap.end()) {
            std::cerr << "Concat layer not found" << std::endl;
            exit(1);
        }
        return concatLayerMap[id];
    }

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale)
    {
        Tensor4D<T> ip(d1, d2, d3, d4);
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        Layer<T>::fakeExecution = true;
        auto &res = this->_forward(ip);
        Layer<T>::fakeExecution = false;
        root = ip.graphNode;
        activation.resize(res.d1, res.d2, res.d3, res.d4);
        
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            auto &inp = node->layer->inputDerivative;
            node->layer->init(inp.d1, inp.d2, inp.d3, inp.d4, scale);
        });

        this->scale = scale;
        generateAddLayerMap();
        generateConcatLayerMap();
    }

    void init(u64 scale)
    {
        Tensor4D<T> ip(0, 0, 0, 0);
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        Layer<T>::fakeExecution = true;
        auto &res = this->_forward(ip);
        Layer<T>::fakeExecution = false;
        root = ip.graphNode;
        
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->initScale(scale);
        });

        this->scale = scale;
        generateAddLayerMap();
        generateConcatLayerMap();
    }

    void zero()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (node->layer->name == "Conv2D" || node->layer->name == "FC") {
                node->layer->getweights().fill(0);
                node->layer->getbias().fill(0);
            }
            else if (node->layer->name == "BatchNorm2dInference") {
                BatchNorm2dInference<T> *bn = (BatchNorm2dInference<T> *) node->layer;
                bn->A.fill(0);
                bn->B.fill(0);
            }
        });
    }

    void setBackend(Backend<T> *b)
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->setBackend(b);
        });
        backend = b;
    }

    Tensor4D<T>& forward(Tensor4D<T> &input)
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->numUsages = 0;
        });
        input.graphNode = root;
        input.graphNode->currTensor = &input;
        if (debug) {
            auto& res = this->_forward(input);
            this->activation.resize(res.d1, res.d2, res.d3, res.d4);
            this->activation.copy(res);
            return this->activation;
        }
        else {
            auto& res = this->_forward(input); // todo: calculate using the generated graph
            this->activation.resize(res.d1, res.d2, res.d3, res.d4);
            this->activation.copy(res);
            return this->activation;
        }
    }

    void optimize()
    {
        backend->optimize(root);
    }

    void load(const std::string weightsFile)
    {
        size_t size_in_bytes = std::filesystem::file_size(weightsFile);
        always_assert(size_in_bytes % 4 == 0); // as it's float
        size_t numParameters = size_in_bytes / 4;
        float *floatWeights = new float[numParameters];
        
        std::ifstream file(weightsFile, std::ios::binary);
        file.read((char*) floatWeights, size_in_bytes);
        file.close();
        u64 scale = this->scale;
        
        size_t wIdx = 0;
        for (auto &node: allNodesInExecutionOrder) {
            auto layer = node->layer;
            if(layer->name.find("Conv2D") != std::string::npos || layer->name.find("FC") != std::string::npos) {
                auto& weights = layer->getweights();
                for (int j = 0; j < weights.d1; j++) {
                    for(int k = 0; k < weights.d2; ++k) {
                        weights(j, k) = i64(floatWeights[wIdx + weights.d2 * j + k] * (1LL << scale));
                    }
                }

                auto wSize = weights.d1 * weights.d2;
                wIdx += wSize;

                auto& bias = layer->getbias();
                if (layer->useBias) {

                    for (int j = 0; j < bias.size; ++j) {
                        bias(j) = i64(floatWeights[wIdx + j] * (1LL << (2*scale)));
                    }

                    wSize = bias.size;
                    wIdx += wSize;
                }
                else
                    bias.fill(0);
            }
            else if (layer->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layer;
                auto channel = bn->A.size;
                auto gammaPtr = floatWeights + wIdx;
                auto betaPtr = floatWeights + wIdx + channel;
                auto meanPtr = floatWeights + wIdx + 2 * channel;
                auto varPtr = floatWeights + wIdx + 3 * channel;
                for (int j = 0; j < channel; ++j) {
                    bn->A(j) = i64((gammaPtr[j] / std::sqrt(varPtr[j])) * (1LL << scale));
                    bn->B(j) = i64((betaPtr[j] - gammaPtr[j] * meanPtr[j] / std::sqrt(varPtr[j])) * (1LL << (2 * scale)));
                }
                wIdx += 4 * channel;
            }
        }

        always_assert(wIdx == numParameters);
        delete[] floatWeights;
    }

    template <typename... Args>
    std::vector<Tensor4D<T> *> collect(Args & ... args)
    {
        std::vector<Tensor4D<T> *> res;
        collectHelper(res, args...);
        return res;
    }

    void collectHelper(std::vector<Tensor4D<T> *> &res, Tensor4D<T> &a)
    {
        res.push_back(&a);
    }

    template <typename... Args>
    void collectHelper(std::vector<Tensor4D<T> *> &res, Tensor4D<T> &a, Args & ... args)
    {
        res.push_back(&a);
        collectHelper(res, args...);
    }

    void add(std::vector<Tensor4D<T> *> &arr, Tensor4D<T> &c)
    {
        if (Layer<T>::fakeExecution) {
            c.graphNode = new LayerGraphNode<T>();
            c.graphNode->layer = new PlaceHolderLayer<T>("Add");
            for (auto &a : arr) {
                c.graphNode->parents.push_back(a->graphNode);
                a->graphNode->children.push_back(c.graphNode);
            }
            c.graphNode->allNodesInExecutionOrderRef = arr[0]->graphNode->allNodesInExecutionOrderRef;
            c.graphNode->allNodesInExecutionOrderRef->push_back(c.graphNode);
            return;
        }

        // check if all tensors have same dimensions
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[i]->d1 != arr[0]->d1 || arr[i]->d2 != arr[0]->d2 || arr[i]->d3 != arr[0]->d3 || arr[i]->d4 != arr[0]->d4) {
                throw std::runtime_error("All tensors must have same dimensions");
            }
        }

        auto cNode = getAddNode(arr);
        cNode->currTensor = &c;
        c.graphNode = cNode;

        for (int i = 0; i < c.d1; ++i) {
            for (int j = 0; j < c.d2; ++j) {
                for (int k = 0; k < c.d3; ++k) {
                    for (int l = 0; l < c.d4; ++l) {
                        c(i, j, k, l) = 0;
                        for (auto &a : arr) {
                            c(i, j, k, l) += a->operator()(i, j, k, l);
                        }
                    }
                }
            }
        }

        for (auto &a : arr) {
            bool gcHappened = a->graphNode->incrementAndGc();
            // if (gcHappened) {
            //     std::cerr << "Output of " << a->graphNode->layer->name << " cleared" << std::endl;
            // }
        }
    }

    Tensor4D<T> add(std::vector<Tensor4D<T> *> &arr)
    {
        Tensor4D<T> c(arr[0]->d1, arr[0]->d2, arr[0]->d3, arr[0]->d4);
        add(arr, c);
        return c;
    }

    template <typename... Args>
    Tensor4D<T> add(Args & ... args)
    {
        auto res = collect(args...);
        return add(res);
    }

    void concat(std::vector<Tensor4D<T> *> &arr, Tensor4D<T> &c)
    {
        if (Layer<T>::fakeExecution) {
            c.graphNode = new LayerGraphNode<T>();
            c.graphNode->layer = new PlaceHolderLayer<T>("Concat");
            for (auto &a : arr) {
                c.graphNode->parents.push_back(a->graphNode);
                a->graphNode->children.push_back(c.graphNode);
            }
            c.graphNode->allNodesInExecutionOrderRef = arr[0]->graphNode->allNodesInExecutionOrderRef;
            c.graphNode->allNodesInExecutionOrderRef->push_back(c.graphNode);
            return;
        }

        // check if all tensors have same dimensions except the last one
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[i]->d1 != arr[0]->d1 || arr[i]->d2 != arr[0]->d2 || arr[i]->d3 != arr[0]->d3) {
                throw std::runtime_error("All tensors must have same dimensions");
            }
        }

        auto cNode = getConcatNode(arr);
        cNode->currTensor = &c;
        c.graphNode = cNode;

        u64 d4 = 0;
        for (auto &a : arr) {
            d4 += a->d4;
        }

        if (c.d1 != arr[0]->d1 || c.d2 != arr[0]->d2 || c.d3 != arr[0]->d3 || c.d4 != d4) {
            throw std::runtime_error("Output tensor must have correct dimensions");
        }

        u64 d4Idx = 0;
        for (auto &a : arr) {
            for (int i = 0; i < c.d1; ++i) {
                for (int j = 0; j < c.d2; ++j) {
                    for (int k = 0; k < c.d3; ++k) {
                        for (int l = 0; l < a->d4; ++l) {
                            c(i, j, k, d4Idx + l) = a->operator()(i, j, k, l);
                        }
                    }
                }
            }
            d4Idx += a->d4;
        }

        for (auto &a : arr) {
            bool gcHappened = a->graphNode->incrementAndGc();
            // if (gcHappened) {
            //     std::cerr << "Output of " << a->graphNode->layer->name << " cleared" << std::endl;
            // }
        }
    }

    Tensor4D<T> concat(std::vector<Tensor4D<T> *> &arr)
    {
        u64 d4 = 0;
        for (auto &a : arr) {
            d4 += a->d4;
        }
        Tensor4D<T> c(arr[0]->d1, arr[0]->d2, arr[0]->d3, d4);
        concat(arr, c);
        return c;
    }

    template <typename... Args>
    Tensor4D<T> concat(Args & ... args)
    {
        auto res = collect(args...);
        return concat(res);
    }
};