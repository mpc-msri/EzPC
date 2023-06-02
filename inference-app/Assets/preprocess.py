from PIL import Image
import numpy as np
import sys
import os
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_arr_from_image(img):
    arr = preprocess(img).unsqueeze(0).cpu().detach().numpy()
    return arr
