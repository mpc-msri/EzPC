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
ext = full_path[full_path.find("."):]
print(ext)

#Code for resizing larger into smaller square images.
target_size = 224
sess = tf.Session()
img = tf.io.read_file(full_path)
if(ext == ".jpg" or ext == ".jpeg"):
    print("Loading JPEG")
    rgb_img = tf.image.decode_jpeg(img, channels = 3)
elif(ext == ".png"):
    print("Loading PNG")
    rgb_img = tf.image.decode_png(img, channels = 3)
else:
    raise ValueError('A very specific bad thing happened.')

h,w,c = rgb_img.eval(session=sess).shape
smaller_dim = h if h < w else w
new_h =  int(h * (target_size/smaller_dim))
new_w =  int(w * (target_size/smaller_dim))
resized_img = tf.cast(tf.image.resize_images(rgb_img, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), tf.uint8)
resized_img = tf.image.resize_image_with_crop_or_pad(resized_img, target_height=target_size, target_width=target_size)
np_rgb_img = resized_img.eval(session=sess)
np_bgr_img = np_rgb_img[:,:,::-1]

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



