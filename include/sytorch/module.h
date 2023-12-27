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
        cNode->layer->inputDerivative.resize(c.d1, c.d2, c.d3, c.d4);

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

    void dumpOrcaModel(std::string name = "Net")
    {
        std::ofstream file("model.h");
        std::string tab = "    ";

        file << "template <typename T>" << std::endl;
        file << "Model<T> " << name << "(int batchSz) {" << std::endl;
        file << tab << "int bw = 64;" << std::endl;
        
        int i = 0;
        for (auto &n : allNodesInExecutionOrder) {
            i += 1;
            int h = n->layer->inputDerivative.d2;
            int w = n->layer->inputDerivative.d3;
            int c = n->layer->inputDerivative.d4;

            file << tab << "auto layer" << (i-1) << " = new ";
            auto &layer = n->layer;
            if (layer->name == "Conv2D") {
                auto convLayer = (Conv2D<T> *)(layer);
                file << "Conv2DLayer<T>(bw, bw, batchSz, " << h << ", " << w << ", " << c << ", " << convLayer->ks << ", " << convLayer->ks << ", " << convLayer->co << ", " << convLayer->padding << ", " << convLayer->padding << ", " << convLayer->padding << ", " << convLayer->padding << ", " << convLayer->stride << ", " << convLayer->stride << ", " << (convLayer->useBias ? "true" : "false") << ", TruncateType::LocalARS, TruncateType::LocalARS, " << (i == 1 ? "false, true" : "true, false") << ");";
            }
            else if (layer->name == "MaxPool2D") {
                auto maxPoolLayer = (MaxPool2D<T> *)(layer);
                std::string bwToUse = (maxPoolLayer->mode == 3 ? "bw - scale" : "bw");
                file << "MaxPool2DLayer<T>(" << bwToUse << ", " << bwToUse << ", bw, batchSz, " << h << ", " << w  << ", " << c  << ", " << maxPoolLayer->ks << ", " << maxPoolLayer->ks << ", " << maxPoolLayer->stride << ", " << maxPoolLayer->stride << ", " << maxPoolLayer->padding << ", " << maxPoolLayer->padding << ", " << maxPoolLayer->padding << ", " << maxPoolLayer->padding << ");"; 
            }
            else if (layer->name == "FC") {
                auto fcLayer = (FC<T> *)(layer);
                file << "FCLayer<T>(bw, bw, batchSz, " << fcLayer->out << ", " << fcLayer->in << ", TruncateType::LocalARS, TruncateType::LocalARS, " << (fcLayer->useBias ? "true, " : "false, ") << (i == 1 ? "false, true" : "true, false") << ");";
            }
            else if (layer->name == "ReLU") {
                auto reluLayer = (ReLU<T> *)(layer);
                if (reluLayer->mode == 0) {
                    file << "ReLULayer<T>(bw, bw, batchSz * " << h * w * c << ");";
                }
                else if (reluLayer->mode == 2) {
                    file << "ReluSignExtendLayer<T>(bw - scale, bw, batchSz * " << h * w * c << ");";
                }
                else if (reluLayer->mode == 3) {
                    file << "ReLULayer<T>(bw - scale, bw - scale, batchSz * " << h * w * c << ");";
                }
            }
            else if (layer->name == "GlobalAvgPool2D") {
                auto avgPoolLayer = (GlobalAvgPool2D<T> *)(layer);
                file << "AvgPool2DLayer<T>(bw, bw - scale, scale, batchSz, " << h << ", " << w << ", " << c << ", " << h << ", " << w << ", 1, 1, 0, 0, 0, 0, TruncateType::LocalARS, TruncateType::LocalARS);";
            }
            else if (layer->name == "AvgPool2D") {
                auto avgPoolLayer = (AvgPool2D<T> *)(layer);
                file << "AvgPool2DLayer<T>(bw, bw - scale, scale, batchSz, " << h << ", " << w << ", " << c << ", " << avgPoolLayer->ks << ", " << avgPoolLayer->ks << ", " << avgPoolLayer->stride << ", " << avgPoolLayer->stride << ", " << avgPoolLayer->padding << ", " << avgPoolLayer->padding << ", " << avgPoolLayer->padding << ", " << avgPoolLayer->padding << ", TruncateType::LocalARS, TruncateType::LocalARS);";
            }
            else if (layer->name == "Add") {
                std::string bws = (layer->mode == 0) ? "bw" : "bw - scale";
                auto parent = n->parents[0];
                std::cerr << "my parent is " << parent->layer->name << std::endl;
                parent->layer->activation.printshape();
                auto ipSize = layer->inputDerivative.d2 * layer->inputDerivative.d3 * layer->inputDerivative.d4;
                file << "AddLayer<T>(" << bws << ", batchSz * " << ipSize << ");";
            }
            else if (layer->name == "BatchNorm2dInference") {
                auto bnLayer = (BatchNorm2dInference<T> *)(layer);
                file << "BatchNormLayer<T>(bw, scale, batchSz * " << h * w << ", " << c << ", TruncateType::LocalLRS, " << (layer->doPreSignExtension ? "true" : "false") << ");";
            }
            else if (layer->name == "Concat") {
                file << "ConcatLayer<T>();";
            }
            else if (layer->name == "Flatten") {
                file << "FlattenLayer<T>();";
            }
            else {
                file << "Layer;";
            }
            
            if (layer->doPreSignExtension) {
                file << "// needs pre sign extension" << std::endl;
            }
            else {
                file << std::endl;
            }
        }

        file << std::endl;
        file << tab << "Model<T> m;" << std::endl;
        file << tab << "m.bw = bw;" << std::endl;
        file << tab << "m.batchSz = batchSz;" << std::endl;
        file << tab << "m.numLayers = " << allNodesInExecutionOrder.size() << ";" << std::endl;
        file << tab << "m.classes = " << activation.d2 << ";" << std::endl;
        file << tab << "m.H = " << allNodesInExecutionOrder[0]->layer->inputDerivative.d2 << ";" << std::endl;
        file << tab << "m.W = " << allNodesInExecutionOrder[0]->layer->inputDerivative.d3 << ";" << std::endl;
        file << tab << "m.C = " << allNodesInExecutionOrder[0]->layer->inputDerivative.d4 << ";" << std::endl;
        file << tab << "m.layers = new GPULayer<T>*[m.numLayers];" << std::endl;
        file << tab << "GPULayer<T>* layers[] = {" << std::endl;
        for (int i = 0; i < allNodesInExecutionOrder.size(); i++) {
            file << tab << tab << "layer" << i << "," << std::endl;
        }
        file << tab << "};" << std::endl;
        file << tab << "memcpy(m.layers, layers, m.numLayers * sizeof(GPULayer<T>*));" << std::endl;

        for(int i = 0; i < allNodesInExecutionOrder.size(); ++i) {
            std::string x = "";
            for(int j = 0; j < allNodesInExecutionOrder[i]->parents.size(); ++j)
            {
                // find index of j'th parent in allNodesInExecutionOrder

                for(int k = 0; k < allNodesInExecutionOrder.size(); ++k)
                {
                    if(allNodesInExecutionOrder[k] == allNodesInExecutionOrder[i]->parents[j])
                    {
                        x += std::to_string(k);
                        break;
                    }
                }
                
                if (j != allNodesInExecutionOrder[i]->parents.size() - 1)
                    x += ", ";
            }
            file << tab << "m.topologicalOrderDescription.push_back({ " << i << ", { " << x << " } });" << std::endl;
        }


        file << tab << "return m;" << std::endl;

        file << "}" << std::endl;
    }

    void dumModelWeightsAsi64(std::string filename)
    {
        std::ofstream dealerFile(filename + "_weights_dealer.dat");
        std::ofstream evalFile(filename + "_weights_evaluator.dat");
        i64 zeroStr = 0;

        for (auto &node: allNodesInExecutionOrder) {
            auto layer = node->layer;
            if(layer->name.find("Conv2D") != std::string::npos || layer->name.find("FC") != std::string::npos) {
                auto& weights = layer->getweights();
                evalFile.write((char*)weights.data, weights.d1 * weights.d2 * sizeof(i64));
                for (int i = 0; i < weights.d1; ++i) {
                    for (int j = 0; j < weights.d2; ++j) {
                        dealerFile.write((char*)&zeroStr, sizeof(i64));
                    }
                }
                auto& bias = layer->getbias();
                if (layer->useBias) {
                    evalFile.write((char*)bias.data, bias.size * sizeof(i64));
                    for (int i = 0; i < bias.size; ++i) {
                        dealerFile.write((char*)&zeroStr, sizeof(i64));
                    }
                }
            }
            else if (layer->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layer;
                auto channel = bn->A.size;
                evalFile.write((char*)bn->A.data, channel * sizeof(i64));
                evalFile.write((char*)bn->B.data, channel * sizeof(i64));
                for (int i = 0; i < channel; ++i) {
                    dealerFile.write((char*)&zeroStr, sizeof(i64));
                }
                for (int i = 0; i < channel; ++i) {
                    dealerFile.write((char*)&zeroStr, sizeof(i64));
                }
            }
        }
        dealerFile.close();
        evalFile.close();
    }
};