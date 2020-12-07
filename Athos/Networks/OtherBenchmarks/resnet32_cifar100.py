

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet32 model for Keras adapted from keras.applications.ResNet50.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import functools
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, Dense, Reshape, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_custom_objects

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TFCompiler'))
import DumpTFMtData

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-4


def identity_building_block(input_tensor, kernel_size, filters, stage, block, training=None):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
                middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: current block label, used for generating layer names
        training: Only used if training keras model with Estimator.  In other
            scenarios it is handled automatically.

    Returns:
        Output tensor for the block.
    """
    filters1, filters2 = filters
    bn_axis=1
    if tf.keras.backend.image_data_format() == 'channels_last':
       bn_axis = 3
    else:
       bn_axis = 1


    x = Conv2D(filters1, kernel_size,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            l2(L2_WEIGHT_DECAY))(input_tensor)
    x = BatchNormalization(axis=bn_axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,fused=True)(
                                                x, training=training)

    x = Activation('approx_activation')(x)

    x = Conv2D(filters2, kernel_size,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            l2(L2_WEIGHT_DECAY))(x)
    x = BatchNormalization(axis=bn_axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,fused=True)(
                                                x, training=training)
    x = add([x, input_tensor])
    x = Activation('approx_activation')(x)
    return x


def conv_building_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), training=None):
    """A block that has a conv layer at shortcut.

    Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
                middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        training: Only used if training keras model with Estimator.  In other
            scenarios it is handled automatically.

    Returns:
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters
    bn_axis=1
    if tf.keras.backend.image_data_format() == 'channels_last':
       bn_axis = 3
    else:
       bn_axis = 1

    x = Conv2D(filters1, kernel_size, strides=strides,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            l2(L2_WEIGHT_DECAY))(input_tensor)
    x = BatchNormalization(axis=bn_axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,fused=True)(
                                                x, training=training)
    x = Activation('approx_activation')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            l2(L2_WEIGHT_DECAY))(x)
    x = BatchNormalization(axis=bn_axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,fused=True)(
                                                x, training=training)

    shortcut = Conv2D(filters2, (1, 1), strides=strides,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=
                                   l2(L2_WEIGHT_DECAY),
                                   bias_regularizer=
                                   l2(L2_WEIGHT_DECAY))(input_tensor)
    shortcut = BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,fused=True)(
                    shortcut, training=training)

    x = add([x, shortcut])
    x = Activation('approx_activation')(x)
    return x


def resnet_block(input_tensor, size, kernel_size, filters, stage, conv_strides=(2, 2), training=None):
    """A block which applies conv followed by multiple identity blocks.

    Arguments:
        input_tensor: input tensor
        size: integer, number of constituent conv/identity building blocks.
        A conv block is applied once, followed by (size - 1) identity blocks.
        kernel_size: default 3, the kernel size of
                middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        conv_strides: Strides for the first conv layer in the block.
        training: Only used if training keras model with Estimator.  In other
            scenarios it is handled automatically.

    Returns:
        Output tensor after applying conv and identity blocks.
    """

    x = conv_building_block(input_tensor, kernel_size, filters, stage=stage,
                            strides=conv_strides, block='block_0',
                            training=training)
    for i in range(size - 1):
        x = identity_building_block(x, kernel_size, filters, stage=stage,
                                    block='block_%d' % (i + 1), training=training)
    return x

def build():
    """Instantiates ResNet32 model """
    # Parameters for Resnet32 on Cifar-100
    num_blocks = 5
    classes = 100

    training = False
    input_shape = (32, 32, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input
    bn_axis = 1
    if tf.keras.backend.image_data_format() == 'channels_last':
       bn_axis = 3
    else:
       bn_axis = 1

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(16, (3, 3),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            l2(L2_WEIGHT_DECAY))(x)
    x = BatchNormalization(axis=bn_axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,fused=True)(
                                                x, training=training)
    x = Activation('approx_activation')(x)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[16, 16],
                                      stage=2, conv_strides=(1, 1), training=training)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[32, 32],
                                      stage=3, conv_strides=(2, 2), training=training)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[64, 64],
                                      stage=4, conv_strides=(2, 2), training=training)

    x = AveragePooling2D(pool_size=(8,8),strides=(1,1),padding="VALID")(x)
    x = Lambda(lambda w: tf.keras.backend.squeeze(w,1))(x)
    x = Lambda(lambda w: tf.keras.backend.squeeze(w,1))(x)
    x = Dense(classes,activation='softmax',
                       kernel_initializer='he_normal',
                       kernel_regularizer=
                       l2(L2_WEIGHT_DECAY),
                       bias_regularizer=
                       l2(L2_WEIGHT_DECAY))(x)

    inputs = img_input
    # Create model.
    model = tf.keras.models.Model(inputs, x)

    return model

def main():
    get_custom_objects().update({'approx_activation': Activation(tf.keras.activations.relu)})
    model = build()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        imgData = np.full((1,32,32,3), 2.3)
        feed_dict={'input_1:0':imgData}
        
        ans = model.predict(imgData)
        print(ans)
        
        output_tensor = None
        gg = tf.get_default_graph()
        for node in gg.as_graph_def().node:
            if node.name == 'dense/Softmax':
                output_tensor = gg.get_operation_by_name(node.name).outputs[0]

        assert(output_tensor is not None)
        optimized_graph_def = DumpTFMtData.save_graph_metadata(output_tensor, sess, feed_dict=feed_dict)

if __name__ == "__main__":
    main()