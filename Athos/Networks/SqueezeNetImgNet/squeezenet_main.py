# Copyright (c) 2017 Andrey Voroshilov

import os, sys
import time
import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from PIL import Image
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

def imread_resize(path):
    img_orig = Image.open(path).convert("RGB")
    img_orig = np.asarray(img_orig)
    
    # NOTE: scipy.misc.imresize is deprecated in > v1.1.0. 
    #   But i cannot find a suitable replacement for this which returns 
    #   exactly the same float value after resizing as scipy.misc.resize.
    #   So, as an alternative, try reinstalling scipy v1.1.0 and then run this code.
    #   Install Scipy v1.1 as : pip3 install scipy==1.1.0
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float) 
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    
def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32
    
# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

all_weights = []

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name, runPrediction=True):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', runPrediction=runPrediction)
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', runPrediction=runPrediction)
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME', runPrediction=runPrediction)
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False, keep_prob=None, runPrediction=True):
    net = {}
    cr_time = time.time()

    x = tf.cast(input_image, get_dtype_tf())

    # Feature extractor
    #####################
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2), runPrediction=runPrediction)
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2', runPrediction=runPrediction)
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3', runPrediction=runPrediction)
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4', runPrediction=runPrediction)
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5', runPrediction=runPrediction)
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6', runPrediction=runPrediction)
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7', runPrediction=runPrediction)
    x = fire_cluster(net, x, preloaded, cluster_name='fire8', runPrediction=runPrediction)
    x = fire_cluster(net, x, preloaded, cluster_name='fire9', runPrediction=runPrediction)
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Dropout [use value of 50% when training]
        # x = tf.nn.dropout(x, keep_prob)
    
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases, runPrediction=runPrediction)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x
        
        # x = tf.nn.softmax(x)
        # net['classifier_actv'] = x
    
    print("Network instance created: %fs" % (time.time() - cr_time))
   
    return net
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1), runPrediction=True):
    global all_weights
    if runPrediction:
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
                            padding=padding)
        x = tf.nn.bias_add(conv, bias)
    else:
        conv = tf.nn.conv2d(input, tf.Variable(tf.constant(0.1,shape=weights.shape)), strides=(1, stride[0], stride[1], 1),
                padding=padding)
        x = tf.nn.bias_add(conv, tf.Variable(tf.constant(0.1,shape=bias.shape)))
    net[name] = x
    all_weights.append(weights)
    all_weights.append(bias)
    return x

def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in', dest='input', help='input file', metavar='INPUT', required=True)
    ps.add_argument('--saveTFMetadata', dest='saveTFMetadata', type=bool, help='bool to indicate if to save metadata', required=False)
    ps.add_argument('--saveImgAndWtData', dest='saveImgAndWtData', type=bool, help='bool to indicate if to save img and model weights', required=False)
    ps.add_argument('--savePreTrainedWeightsFloat', dest='savePreTrainedWeightsFloat', type=bool, help='bool to indicate if to save model weights float', required=False)
    ps.add_argument('--savePreTrainedWeightsInt', dest='savePreTrainedWeightsInt', type=bool, help='bool to indicate if to save model weights int', required=False)
    ps.add_argument('--saveImgAndWeightsSeparately', dest='saveImgAndWeightsSeparately', type=bool, help='bool to indicate if to save image and model weights int separately', required=False)
    ps.add_argument('--scalingFac', dest='scalingFac', type=int, help='scalingFac', default=15, required=False)
    return ps

def main():
    import time

    parser = build_parser()
    options = parser.parse_args()

    # Loading image
    img_content, orig_shape = imread_resize(options.input)
    img_content_shape = (1,) + img_content.shape

    # Loading ImageNet classes info
    classes = []
    with open('synset_words.txt', 'r') as classes_file:
        classes = classes_file.read().splitlines()

    # Loading network
    data, sqz_mean = load_net('./PreTrainedModel/sqz_full.mat')

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    g = tf.Graph()
    
    # 1st pass - simple classification
    with g.as_default(), tf.Session(config=config) as sess:
        # Building network
        image = tf.placeholder(dtype=get_dtype_tf(), shape=img_content_shape, name="image_placeholder")
        # keep_prob = tf.placeholder(get_dtype_tf())
        keep_prob = 0.0
        saveTFMetadata = False
        if options.saveTFMetadata:
            saveTFMetadata = options.saveTFMetadata
        sqznet = net_preloaded(data, image, 'max', True, keep_prob, runPrediction=not(saveTFMetadata))
        final_class = tf.argmax(sqznet['classifier_pool'],3)

        sess.run(tf.global_variables_initializer())
        imageData = [preprocess(img_content, sqz_mean)]

        feed_dict = {image: imageData}

        output_tensor = None
        gg = tf.get_default_graph()
        for node in gg.as_graph_def().node:
            if node.name == 'ArgMax': # Final activation, not the argmax
                output_tensor = gg.get_operation_by_name(node.name).outputs[0]
        assert(output_tensor is not None)
        if options.saveTFMetadata:
            optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict)
        else:
            # Classifying
            sqz_class = final_class.eval(feed_dict)[0][0][0]

            print(sqz_class)
            # Outputting result
            print("\nclass: [%d] '%s'" % (sqz_class, classes[sqz_class]))

            if options.savePreTrainedWeightsInt:
                DumpTFMtData.dumpTrainedWeightsInt(sess, all_weights, 'SqNet_weights.inp', options.scalingFac, 'w', alreadyEvaluated=True)
            if options.savePreTrainedWeightsFloat:
                DumpTFMtData.dumpTrainedWeightsFloat(sess, all_weights, 'SqNet_weights_float.inp', 'w', alreadyEvaluated=True)
            if options.saveImgAndWtData:
                DumpTFMtData.dumpImgAndWeightsDataSeparate(sess, imageData, all_weights, 'SqNet_img.inp', 'SqNet_weights.inp', options.scalingFac, alreadyEvaluated=True)

if __name__ == '__main__':
    main()