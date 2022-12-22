from PIL import Image
import numpy as np
import sys
import os
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_arr_from_image(path):
    img = Image.open(path)
    arr = preprocess(img).unsqueeze(0).cpu().detach().numpy()
    print(arr.shape)
    return arr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_image.py abc.jpg")
    img_path = sys.argv[1]
    arr = get_arr_from_image(img_path)
    npy_path = os.path.splitext(img_path)[0] + ".npy"
    np.save(npy_path, arr)