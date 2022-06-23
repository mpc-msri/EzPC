In this contest, you will be asked to infer a random selection of images from the [ImageNet dataset](https://www.image-net.org/). You can use [this](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) or any dataset of your choice for training purposes.

## Input
Your input will be an image, which you need to convert to a Numpy array of desired shape ( the shape must match what you define in the model config.json i.e. the input that your model takes) using a pre-processing python script that you can upload. Below is an example:


    # Example function to pre-process input image.
    import sys
    import onnx
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import cv2

    def preprocess(img):
        img = img / 255.
        img = cv2.resize(img, (256, 256))
        h, w = img.shape[0], img.shape[1]
        y0 = (h - 224) // 2
        x0 = (w - 224) // 2
        img = img[y0 : y0+224, x0 : x0+224, :]
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, axes=[2, 0, 1])
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img


    # These arguments will be provided to this python script in the command line
    input = sys.argv[1]
    output = sys.argv[2]

    with Image.open(input) as img:
            img = np.array(img.convert('RGB'))
            img = preprocess(img)
            np.save(output, img)

## Output
Your model must output a 1x1000 array. The elements are the ordered probabilities of the image belonging to the particular class in the ImageNet dataset. The classes can be accessed [here](/static/contests/objects/imagenet_classes.txt), for your reference.