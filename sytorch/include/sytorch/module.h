#include <sytorch/tensor.h>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>

template <typename T>
class SytorchModule {
public:

    Tensor<T> activation;
    Backend<T> *backend = new ClearText<T>;
    LayerGraphNode<T> *root = nullptr;
    bool debug = true;
    u64 scale;

    std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder;
    const std::vector<std::string> functionalLayers = {"Add", "Concat", "GeLU", "SoftMax", "Split", "View", "Transpose", "_MatMul"};
    static std::map<std::string, LayerGraphNode<T> *> functionalLayerMap;

public:

    virtual Tensor<T>& _forward(Tensor<T> &input) = 0;

    SytorchModule() : activation({}), allNodesInExecutionOrder(0)
    {

    }

    void generateFunctionalLayerMap()
    {
        functionalLayerMap.clear();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (std::find(functionalLayers.begin(), functionalLayers.end(), node->layer->name) != functionalLayers.end()) {
                std::string id = node->layer->name;
                for(auto& parent: node->parents) {
                    id += "|" + std::to_string((uint64_t)(parent));
                }
                id = id + "|" + node->layer->paramstring;
                // make sure it already doesn't exist
                always_assert(functionalLayerMap.find(id) == functionalLayerMap.end());
                functionalLayerMap[id] = node;
            }
        });
    }

    template <typename... Args>
    LayerGraphNode<T> *getFunctionalNode(const std::string &layerName, std::vector<Tensor<T> *> ips, Args ... args)
    {
        std::string id = layerName;
        for(auto& ip: ips) {
            id += "|" + std::to_string((uint64_t)(ip->graphNode));
        }
        id = id + "|" + paramstring(args...);
        if (functionalLayerMap.find(id) == functionalLayerMap.end()) {
            std::cerr << "Layer not found" << std::endl;
            exit(1);
        }
        return functionalLayerMap[id];
    }

    template <typename LayerType, typename... Args>
    Tensor<T>& functionalGraphGen(std::vector<Tensor<T> *> arr, Args ... args)
    {
        for (auto &a : arr) {
            always_assert(a->graphGenMode);
        }
        auto layer = new LayerType(args...);
        layer->paramstring = paramstring(args...);
        return layer->forward(arr);
    }

    void genGraphAndExecutionOrder()
    {
        Tensor<T> ip({});
        ip.graphGenMode = true;
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        auto &res = this->_forward(ip);
        ip.graphGenMode = false;
        root = ip.graphNode;
    }

    void init(u64 scale)
    {
        genGraphAndExecutionOrder();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->initScale(scale);
        });

        this->scale = scale;
        generateFunctionalLayerMap();
    }

    void zero()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->getweights().zero();
            node->layer->getbias().zero();
        });
    }

    void setBackend(Backend<T> *b)
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->setBackend(b);
        });
        backend = b;
    }

    Tensor<T>& forward(Tensor<T> &input)
    {
        if (input.graphGenMode) {
            return this->_forward(input);
        }

        if (input.graphNode == nullptr) { // when the module is a top level module
            topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
                node->numUsages = 0;
            });
            input.graphNode = root;
            input.graphNode->currTensor = &input;
        }
        if (debug) {
            auto& res = this->_forward(input);
            this->activation.resize(res.shape);
            this->activation.copy(res);
            return this->activation;
        }
        else {
            auto& res = this->_forward(input); // todo: calculate using the generated graph
            this->activation.resize(res.shape);
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
            if (layer->name == "BatchNormInference") {
                auto bn = (BatchNormInference<T>*) layer;
                auto channel = bn->A.d1;
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
            else {
                auto weights = layer->getweights();
                for (u64 j = 0; j < weights.size; j++) {
                    weights.data[j] = i64(floatWeights[wIdx + j] * (1LL << scale));
                }

                wIdx += weights.size;

                auto bias = layer->getbias();
                if (layer->useBias) {

                    for (u64 j = 0; j < bias.size; ++j) {
                        bias.data[j] = i64(floatWeights[wIdx + j] * (1LL << (2*scale)));
                    }

                    wIdx += bias.size;
                }
                else {
                    bias.zero();
                }
            }
        }

        always_assert(wIdx == numParameters);
        delete[] floatWeights;
    }

    Tensor<T>& add(std::vector<Tensor<T> *> &arr)
    {
        if (arr[0]->graphGenMode) {
            auto &c = functionalGraphGen<Add<T>>(arr);
            return c;
        }

        auto cNode = getFunctionalNode("Add", arr);
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    template <typename... Args>
    Tensor<T>& add(Args & ... args)
    {
        auto res = collect(args...);
        return add(res);
    }

    Tensor<T>& concat(std::vector<Tensor<T> *> &arr)
    {
        if (arr[0]->graphGenMode) {
            auto &c = functionalGraphGen<Concat<T>>(arr);
            return c;
        }

        auto cNode = getFunctionalNode("Concat", arr);
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    template <typename... Args>
    Tensor<T> concat(Args & ... args)
    {
        auto res = collect(args...);
        return concat(res);
    }

    Tensor<T>& gelu(Tensor<T> &a)
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<GeLU<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("GeLU", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T>& softmax(Tensor<T> &a)
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<SoftMax<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("SoftMax", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T>& split(Tensor<T> &a, u64 n_splits) 
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<Split<T>>({&a}, n_splits);
            return c;
        }

        auto cNode = getFunctionalNode("Split", {&a}, n_splits);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T>& view(Tensor<T> &a, u64 idx)
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<View<T>>({&a}, idx);
            return c;
        }

        auto cNode = getFunctionalNode("View", {&a}, idx);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T>& transpose(Tensor<T> &a)
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<Transpose<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("Transpose", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T>& matmul(Tensor<T> &a, Tensor<T> &b)
    {
        if (a.graphGenMode) {
            auto &c = functionalGraphGen<_MatMul<T>>({&a, &b});
            return c;
        }

        auto cNode = getFunctionalNode("_MatMul", {&a, &b});
        std::vector<Tensor<T> *> arr = {&a, &b};
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    void train()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->train();
        });
    }

    void eval()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->eval();
        });
    }
};

template <typename T>
std::map<std::string, LayerGraphNode<T> *> SytorchModule<T>::functionalLayerMap = std::map<std::string, LayerGraphNode<T> *>();
