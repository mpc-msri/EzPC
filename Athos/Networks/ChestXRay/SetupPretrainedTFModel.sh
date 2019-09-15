#!/bin/bash

kerasModelLink="https://chestxray.blob.core.windows.net/chestxraytutorial/tutorial_xray/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5"
axel -a -n 3 -c --output PreTrainedModel/KerasModel "$kerasModelLink"
cd PreTrainedModel
git clone https://github.com/amir-abdi/keras_to_tensorflow
cd keras_to_tensorflow
echo -e "Starting keras to TF model conversion....\n"
python3 keras_to_tensorflow.py --output_meta_ckpt=True --save_graph_def=True --input_model="../KerasModel/chexray_14_weights_712split_epoch_054_val_loss_191.2588.hdf5" --output_model="../TFModel/model.pb"
