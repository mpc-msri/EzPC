import tensorflow as tf
import sys
import numpy as np

if(len(sys.argv) == 3):
    ground_label = sys.argv[1]
    image_path = sys.argv[2]
    print(ground_label, image_path)
    full_path = "val/" + ground_label + "/" + image_path
elif(len(sys.argv) == 2):
    full_path = sys.argv[1]

print(full_path)

img = tf.io.read_file(full_path)
rgb_img = tf.image.decode_jpeg(img, channels = 3)
resized_img = tf.image.resize(rgb_img, [224, 224])
sess = tf.Session()
np_rgb_img = resized_img.eval(session=sess)
# np_bgr_img = np_rgb_img[:,:,::-1]
np_bgr_img = np_rgb_img
print(np_bgr_img.shape)
np_bgr_img = np.transpose(np_bgr_img, (1, 2, 0))
print(np_bgr_img.shape)
np_bgr_img = np.transpose(np_bgr_img, (1, 2, 0))
print(np_bgr_img.shape)
np.save("debug/cov7/prep_input_covid_BGR.npy", np_bgr_img)

#testing
# b = np.zeros([2, 2, 3], dtype=np.uint8)
# b[:,:,0] = 0 #r
# b[:,:,1] = 1 #g
# b[:,:,2] = 2 #b
# print(b.data.tolist())
# bgr = b[:,:,::-1]
# print(bgr.data.tolist())



