"""
Authors: Saksham Gupta.
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
