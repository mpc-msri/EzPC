#pragma once

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class GPUModel
        {
        public:
            std::vector<GPULayer<T> *> layers;
            int classes;
            int batchSz, inpSz;

            void initWeights(std::string weightsFile, bool floatWeights = false)
            {
                std::cout << weightsFile << std::endl;
                if (weightsFile.compare("") != 0)
                {
                    size_t wSize;
                    auto weights = readFile(weightsFile, &wSize, false);
                    auto tmpWeights = weights;
                    for (int i = 0; i < layers.size(); i++)
                    {
                        layers[i]->initWeights(&tmpWeights, floatWeights);
                    }
                    free(weights);
                }
            }

            void setTrain(bool useMomentum)
            {
                for (int i = 0; i < layers.size(); i++)
                    layers[i]->setTrain(useMomentum);
            }

            void dumpWeights(std::string filename)
            {
                std::ofstream f(filename);
                if (!f)
                {
                    std::cerr << "can't open output file=" << filename << std::endl;
                    assert(0);
                }
                for (int i = 0; i < layers.size(); i++)
                {
                    layers[i]->dumpWeights(f);
                }
                f.flush();
                f.close();
            }
        };

    }
}