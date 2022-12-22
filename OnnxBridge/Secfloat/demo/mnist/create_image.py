import numpy as np
import idx2numpy
from PIL import Image


imagefile = "mnist/t10k-images-idx3-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)
temp = imagearray[1]
temp = np.reshape(temp, (28, 28))
temp = temp * 255
im = Image.fromarray(temp)
im.save("input.jpg")
