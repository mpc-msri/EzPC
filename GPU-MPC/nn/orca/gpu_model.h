// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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