# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/sample/yolo.py
# Author: FanJH
# Description: 
#############################################
import numpy as np
import tensorflow as tf
from config import cfg

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)

    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type="leaky"):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    # return tf.image.resize(input_layer, (tf.shape(input_layer)[1] * 2, tf.shape(input_layer)[2] * 2), method='nearest')
    return tf.image.resize(input_layer, (int(input_layer.shape[1]) * 2, int(input_layer.shape[2]) * 2), method='nearest')

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def darknet53(input_data):

    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def cspdarknet53(input_data,activate_type="leaky"):
    input_data = convolutional(input_data,(3,3,3,32),activate_type=activate_type)
    #downsample1
    input_data = convolutional(input_data,(3,3,32,64),downsample=True,activate_type=activate_type)
    route = input_data
    route = convolutional(route,(1,1,64,64),activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,64,64),activate_type=activate_type)
    for i in range(1):
        input_data = residual_block(input_data,64,32,64,activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,64,64),activate_type=activate_type)
    input_data = tf.concat([input_data,route],-1)
    input_data = convolutional(input_data,(1,1,128,64),activate_type=activate_type)
    #downsample2
    input_data = convolutional(input_data,(3,3,64,128),downsample=True,activate_type=activate_type)
    route = input_data
    route = convolutional(route,(1,1,128,64),activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,128,64),activate_type=activate_type)
    for i in range(2):
        input_data = residual_block(input_data,64,64,64,activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,64,64),activate_type=activate_type)
    input_data = tf.concat([input_data,route],-1)
    input_data = convolutional(input_data,(1,1,128,128),activate_type=activate_type)
    #downsample3
    input_data = convolutional(input_data,(3,3,128,256),downsample=True,activate_type=activate_type)
    route = input_data
    route = convolutional(route,(1,1,256,128),activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,256,128),activate_type=activate_type)
    for i in range(8):
        input_data = residual_block(input_data,128,128,128,activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,128,128),activate_type=activate_type)
    input_data = tf.concat([input_data,route],-1)
    input_data = convolutional(input_data,(1,1,256,256),activate_type=activate_type)
    route_1 = input_data
    #downsample4
    input_data = convolutional(input_data,(3,3,256,512),downsample=True,activate_type=activate_type)
    route = input_data
    route = convolutional(route,(1,1,512,256),activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,512,256),activate_type=activate_type)
    for i in range(8):
        input_data = residual_block(input_data,256,256,256,activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,256,256),activate_type=activate_type)
    input_data = tf.concat([input_data,route],-1)
    input_data = convolutional(input_data,(1,1,512,512),activate_type=activate_type)
    route_2 = input_data
    #downsample5
    input_data = convolutional(input_data,(3,3,512,1024),downsample=True,activate_type=activate_type)
    route = input_data
    route = convolutional(route,(1,1,1024,512),activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,1024,512),activate_type=activate_type)
    for i in range(4):
        input_data = residual_block(input_data,512,512,512,activate_type=activate_type)
    input_data = convolutional(input_data,(1,1,512,512),activate_type=activate_type)
    input_data = tf.concat([input_data,route],-1)
    input_data = convolutional(input_data,(1,1,1024,1024),activate_type=activate_type)
    #convset1
    input_data = convolutional(input_data,(1,1,1024,512))
    input_data = convolutional(input_data,(3,3,512,1024))
    input_data = convolutional(input_data,(1,1,1024,512))
    #spp module
    pool_data1 = tf.keras.layers.MaxPooling2D(pool_size=13,strides=1,padding="same")(input_data)
    pool_data2 = tf.keras.layers.MaxPooling2D(pool_size=9,strides=1,padding="same")(input_data)
    pool_data3 = tf.keras.layers.MaxPooling2D(pool_size=5,strides=1,padding="same")(input_data)
    pool_data = tf.concat([pool_data1,pool_data2],-1)
    pool_data = tf.concat([pool_data,pool_data3],-1)
    input_data = tf.concat([pool_data,input_data],-1)
    #convset2
    input_data = convolutional(input_data,(1,1,2048,512))
    input_data = convolutional(input_data,(3,3,512,1024))
    input_data = convolutional(input_data,(1,1,1024,512))

    return route_1,route_2,input_data

def YOLOv3(input_layer,num_class):
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))

    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(num_class + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1,  512,  256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(num_class + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(num_class +5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer,num_class):
    route_1,route_2,conv = cspdarknet53(input_layer,activate_type="leaky")
    #PAN
    route = conv
    conv = convolutional(conv,(1,1,512,256))
    conv = upsample(conv)
    route_2 = convolutional(route_2,(1,1,512,256))
    conv = tf.concat([route_2,conv],-1)
    #convset3
    conv = convolutional(conv,(1,1,512,256))
    conv = convolutional(conv,(3,3,256,512))
    conv = convolutional(conv,(1,1,512,256))
    conv = convolutional(conv,(3,3,256,512))
    conv = convolutional(conv,(1,1,512,256))

    route_2 = conv
    conv = convolutional(conv,(1,1,256,128))
    conv = upsample(conv)
    route_1 = convolutional(route_1,(1,1,256,128))
    conv = tf.concat([route_1,conv],-1)
    #convset4
    conv = convolutional(conv,(1,1,256,128))
    conv = convolutional(conv,(3,3,128,256))
    conv = convolutional(conv,(1,1,256,128))
    conv = convolutional(conv,(3,3,128,256))
    conv = convolutional(conv,(1,1,256,128))

    route_1 = conv
    conv = convolutional(conv,(3,3,128,256))
    conv_sbbox = convolutional(conv,(1,1,256,3*(num_class+5)),activate=False,bn=False)
    conv = convolutional(route_1,(3,3,128,256),downsample=True)
    conv = tf.concat([conv,route_2],-1)
    #convset5
    conv = convolutional(conv,(1,1,512,256))
    conv = convolutional(conv,(3,3,256,512))
    conv = convolutional(conv,(1,1,512,256))
    conv = convolutional(conv,(3,3,256,512))
    conv = convolutional(conv,(1,1,512,256))

    route_2 = conv
    conv = convolutional(conv,(3,3,256,512))
    conv_mbbox = convolutional(conv,(1,1,512,3*(num_class+5)),activate=False,bn=False)
    conv = convolutional(route_2,(3,3,256,512),downsample=True)
    conv = tf.concat([conv,route],-1)
    #convset6
    conv = convolutional(conv,(1,1,1024,512))
    conv = convolutional(conv,(3,3,512,1024))
    conv = convolutional(conv,(1,1,1024,512))
    conv = convolutional(conv,(3,3,512,1024))
    conv = convolutional(conv,(1,1,1024,512))

    conv = convolutional(conv,(3,3,512,1024))
    conv_lbbox = convolutional(conv,(1,1,1024,3*(num_class+5)),activate=False,bn=False)

    return [conv_sbbox,conv_mbbox,conv_lbbox]