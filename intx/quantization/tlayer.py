# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/quantization/tlayer.py
# Author: FanJH
# Description: Quantized Layer
#############################################

import tensorflow as tf
from intx.quantization.param import QParam
    
@tf.custom_gradient
def STE_ROUND(x):
    return tf.round(x),lambda dy:dy
    
def FakeQuant(tensor,qparam,beta=0.99):
    if tensor.shape[0] is None:
        return tensor
    qparam.update(tensor,beta)
    return tensor
    qparam.freeze()
    if qparam.scale is None or qparam.zero_point is None:
        return tensor
    if qparam.scale == 0.:
        return tensor
        
    q_tensor = tensor/qparam.scale + qparam.zero_point
    q_tensor = STE_ROUND(q_tensor)
    q_tensor = tf.clip_by_value(q_tensor,qparam.qmin,qparam.qmax)
    tensor = qparam.scale * (q_tensor-qparam.zero_point)

    return tensor


class QPlaceholder(tf.keras.layers.Layer):
    def __init__(self,strategy="minmax",num_bits=8,signed=False):
        super(QPlaceholder,self).__init__()
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self,x):
        if self.inferring:
            x = self.qo.quantize_tensor(x) - self.qo.zero_point
            x = tf.cast(x,tf.float32)
        
        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)
            
        return x

class QIdentity(tf.keras.layers.Layer):
    def __init__(self,strategy="minmax",num_bits=8,signed=False):
        super(QIdentity,self).__init__()
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self,x):
        if self.inferring:
            x += self.qo.zero_point
            x = self.qo.dequantize_tensor(x)
            x = tf.cast(x,tf.float32)

        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)
        
        return x

class QConv2D(tf.keras.layers.Conv2D):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        kernel_initializer = tf.keras.initializers.constant(node["filter"])
        if "bias" in node.keys():
            bias_initializer = tf.keras.initializers.constant(node["bias"])
        else:
            bias_initializer = None
        super(QConv2D,self).__init__(filters=node["filter"].shape[-1],
                                    kernel_size=node["filter"].shape[:2],
                                    strides=node["strides"][1:3],
                                    padding=node["padding"],
                                    use_bias= True if "bias" in node.keys() else False,
                                    kernel_initializer = kernel_initializer,
                                    bias_initializer = bias_initializer
                                    )
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
        self.pad = None
        if "paddings" in node.keys():
            self.pad = node["paddings"]
        
    def call(self,x):
        if self.pad is not None:
            x = tf.pad(x,paddings=self.pad)
        if not self.inferring:
            x = tf.nn.conv2d(x,self.kernel,self.strides,self.padding.upper())
            if self.use_bias:
                x = tf.nn.bias_add(x,self.bias)
        else:
            M0 = self.M0[0]
            n = self.n[0]
            x = tf.nn.conv2d(x,self.kernel,self.strides,self.padding.upper())
            x = tf.cast(x,tf.int32)
            if self.use_bias:
                x = tf.nn.bias_add(x,self.bias)
            x = x * M0 + 2**(n-1)
            x = tf.bitwise.right_shift(x,n)
            x = tf.cast(x,tf.float32)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QBatchNorm(tf.keras.layers.BatchNormalization):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        super(QBatchNorm,self).__init__(
                                        epsilon=node["epsilon"], 
                                        beta_initializer=tf.keras.initializers.Constant(node["offset"]), 
                                        gamma_initializer=tf.keras.initializers.Constant(node["scale"]), 
                                        moving_mean_initializer=tf.keras.initializers.Constant(node["mean"]), 
                                        moving_variance_initializer=tf.keras.initializers.Constant(node["variance"])
                                        )
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self,x):
        if not self.inferring:

            x = tf.nn.batch_normalization(x,self.moving_mean,self.moving_variance,self.beta,self.gamma,self.epsilon)

        else:
            M0 = self.M0[0]
            n = self.n[0]
            x = tf.multiply(x,self.gamma)
            x = tf.cast(x,tf.int32)
            x = tf.nn.bias_add(x,self.beta)
            x = x * M0 + 2**(n-1)
            x = tf.bitwise.right_shift(x,n)
            x = tf.cast(x,tf.float32)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QRelu(tf.keras.layers.ReLU):
    def __init__(self, node,strategy="minmax",num_bits=8,signed=False):
        super(QRelu,self).__init__()
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self, x):
        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)
        x = tf.nn.relu(x)

        return x

class QLeaky(tf.keras.layers.LeakyReLU):
    def __init__(self, node,strategy="minmax",num_bits=8,signed=False):
        super(QLeaky,self).__init__(alpha=node["alpha"])
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
    
    def call(self, x):
        if not self.inferring:
            x = tf.nn.leaky_relu(x,alpha=self.alpha)
        else:
            M0 = self.M0[0]
            n = self.n[0]
            x_neg = tf.where(tf.greater_equal(x,0),0,x)
            x_neg = tf.cast(x_neg*self.weight[1],tf.int32)
            x_pos = tf.where(tf.greater_equal(x,0),x,0)
            x_pos = tf.cast(x_pos*self.weight[0],tf.int32)
            x = x_pos + x_neg
            x = x * M0 + 2**(n-1)
            x = tf.bitwise.right_shift(x,n)
            x = tf.cast(x,tf.float32)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)
        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QAdd(tf.keras.layers.Add):
    def __init__(self, strategy="minmax",num_bits=8,signed=False):
        super(QAdd,self).__init__()
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
    
    def call(self, xlist):
        if len(xlist) != 2:
            raise ValueError('A `QAdd` layer should be called on input list at 2 inputs')
        x1,x2 = xlist
        if not self.inferring:
            x = tf.add(x1,x2)
        else:
            M0_1,M0_2 = self.M0
            n_1,n_2 = self.n
            x1 = tf.cast(x1,tf.int32)
            x1 = x1 * M0_1 + 2**(n_1-1)
            x1 = tf.bitwise.right_shift(x1,n_1)
            x1 = tf.cast(x1,tf.float32)

            x2 = tf.cast(x2,tf.int32)
            x2 = x2 * M0_2 + 2**(n_2-1)
            x2 = tf.bitwise.right_shift(x2,n_2)
            x2 = tf.cast(x2,tf.float32)

            x = tf.add(x1,x2)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QCat(tf.keras.layers.Concatenate):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        super(QCat,self).__init__(axis=node["axis"])
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
    
    def call(self, xlist):
        if len(xlist) != 2:
            raise ValueError('A `QCat` layer should be called on input list at 2 inputs')
        x1,x2 = xlist
        if not self.inferring:
            x = tf.concat([x1,x2],axis=self.axis)
        else:
            M0_1,M0_2 = self.M0
            n_1,n_2 = self.n
            x1 = tf.cast(x1,tf.int32)
            x1 = x1 * M0_1 + 2**(n_1-1)
            x1 = tf.bitwise.right_shift(x1,n_1)
            x1 = tf.cast(x1,tf.float32)

            x2 = tf.cast(x2,tf.int32)
            x2 = x2 * M0_2 + 2**(n_2-1)
            x2 = tf.bitwise.right_shift(x2,n_2)
            x2 = tf.cast(x2,tf.float32)

            x = tf.concat([x1,x2],self.axis)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QDense(tf.keras.layers.Dense):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        kernel_initializer = tf.keras.initializers.constant(node["weight"])
        if "bias" in node.keys():
            bias_initializer = tf.keras.initializers.constant(node["bias"])
        else:
            bias_initializer = None
        super(QDense,self).__init__(node["weight"].shape[-1], 
                        activation=None, 
                        use_bias=True if "bias" in node.keys() else False, 
                        kernel_initializer=kernel_initializer, 
                        bias_initializer=bias_initializer
                        )
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
        
    def call(self,x):
        if not self.inferring:
            x = tf.matmul(x,self.kernel)
            if self.use_bias:
                x = tf.nn.bias_add(x,self.bias)
        else:
            M0 = self.M0[0]
            n = self.n[0]
            x = tf.matmul(x,self.kernel)
            x = tf.cast(x,tf.int32)
            if self.use_bias:
                x = tf.nn.bias_add(x,self.bias)
            x = x * M0 + 2**(n-1)
            x = tf.bitwise.right_shift(x,n)
            x = tf.cast(x,tf.float32)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            x = FakeQuant(x,self.qo)

        return x

class QUpSample(tf.keras.layers.UpSampling2D):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        super().__init__(size=node["size"],interpolation=node["interpolation"])
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self, x):
        x = tf.image.resize(x,self.size,method=self.interpolation)
        if self.inferring:
            x = tf.cast(tf.round(x),tf.float32)

        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)
            
        return x

class QMaxPool(tf.keras.layers.MaxPooling2D):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        super(QMaxPool,self).__init__()
        self.pool_size = node["size"]
        self.strides = node["strides"]
        self.padding = node["padding"]
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)
    
    def call(self,x):
        
        x = tf.nn.max_pool2d(x,self.pool_size,self.strides,self.padding)

        if self.inferring:
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)

        return x 
        
class QAvgPool(tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self,node,strategy="minmax",num_bits=8,signed=False):
        super(QAvgPool,self).__init__()
        pool_size = None
        if "pool_size" in node.keys():
            pool_size = node["pool_size"]
        if pool_size is None:
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
        else:
            self.pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size,strides=node["stride"],padding=node["padding"])
        self.inferring = False
        self.qo = QParam(strategy=strategy,num_bits=num_bits,signed=signed)

    def call(self,x):
        if not self.inferring and hasattr(self,"qo"):
            self.qo.update(x)
        x = self.pool(x)
        if self.inferring:
            x = tf.round(x)
            x = tf.clip_by_value(x,self.qo.qmin-self.qo.zero_point,self.qo.qmax-self.qo.zero_point)

        return x  