#Prepare imagenet validation set 

ImageNetValidationSetUrl="http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar"
ImageNetValidationSetBBoxUrl="http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_val_v3.tgz"
ImageNetValidationSetSynSetLabels="https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt"

axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetUrl" 
axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetBBoxUrl" 
axel -a -n 3 -c --output ImageNet_ValData "$ImageNetValidationSetSynSetLabels"
cd ImageNet_ValData
mkdir img
tar -xvf ILSVRC2012_img_val.tar --directory=img
tar -xvzf ILSVRC2012_bbox_val_v3.tgz
mv val bbox
cd ..
python3 Convert_WnId_To_TrainId.py

