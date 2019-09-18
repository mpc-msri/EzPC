This folder contains code for logistic regression on the MNIST dataset. Parts of the code have been taken from [this github repo](https://github.com/aymericdamien/TensorFlow-Examples/) and modified accordingly for our purposes.

## Setup
- First train the model using `python3 LogisticRegressionTrain.py`. This will place the trained model in `./TrainedModel` folder.
- Next run inference using `python3 LogisticRegressionInfer.py 0`, where `0` can be replaced with any MNIST image number. This command will also dump the required metadata required for further compilation.
