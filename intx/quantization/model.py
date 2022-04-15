# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/quantization/model.py
# Author: FanJH
# Description: 
#############################################
import tensorflow as tf
import intx.quantization.tlayer as L
from intx.quantization.param import QParam
from intx.quantization.common import quantize_bias, searchM0

class TFModel():
    def __init__(self,mode,strategy,num_bits,signed):
        super(TFModel,self).__init__()
        self.mode = mode
        self.strategy = strategy
        self.num_bits = num_bits
        self.signed = signed
        self.model = None
        self.freezed = False

    def build(self,input_layer,nodes):
        identity_results = []
        tensors = {-1:input_layer}
        layer_count = [0]
        params = [self.strategy,self.num_bits,self.signed]
        bfs_build(params,0,nodes,tensors,identity_results)

        self.model = tf.keras.Model(input_layer,identity_results)
        self.model(input_layer)

    def __call__(self,inputs,training=True):
        if self.freezed:
            raise ValueError("model have been freezed, can't training or calibration.")

        if self.model == None:
            raise Exception("model is not constructed yet.")
        predict = self.model(inputs,training=training)

        return predict

    def calibrate(self,dataset=None,sample_N=4096):
        if dataset is not None:
            for ind,data in enumerate(dataset):
                image,_ = data
                if ind*image.shape[0] >= sample_N:
                    break
                self.model(image,training=False)
                tf.print("%d/%d calibrate..."%((ind+1)*image.shape[0],sample_N))
        
        for ind,layer in enumerate(self.model.layers):
            inputs = layer.inbound_nodes[0].inbound_layers
            qi_list = []
            if isinstance(inputs,list):
                qi_list = [l.qo for l in inputs if hasattr(l,"qo")]
            else:
                qi_list = [inputs.qo if hasattr(inputs,"qo") else None]
                
            if "placeholder" in layer.name:
                freeze(layer)
            elif "conv2d" in layer.name:
                freeze(layer,qi_list,self.strategy,self.num_bits,self.signed)
            elif "batch_norm" in layer.name:
                freeze_bn(layer,qi_list[0],self.strategy,self.num_bits,self.signed)
            elif "relu" in layer.name:
                freeze(layer)
            elif "leaky" in layer.name:
                freeze_leaky(layer,qi_list[0],self.strategy,self.num_bits,self.signed)
            elif "max_pool" in layer.name:
                freeze(layer,qi_list,self.strategy,self.num_bits,self.signed)
            elif "avg_pool" in layer.name:
                freeze(layer)
            elif "dense" in layer.name:
                freeze(layer,qi_list,self.strategy,self.num_bits,self.signed)
            elif "add" in layer.name:
                freeze(layer,qi_list,self.strategy,self.num_bits,self.signed)
            elif "cat" in layer.name:
                freeze(layer,qi_list,self.strategy,self.num_bits,self.signed)
            elif "up_sample" in layer.name:
                freeze(layer)
            elif "identity" in layer.name:
                freeze(layer)

        self.freezed = True
        for layer in self.model.layers:
            if hasattr(layer,"inferring"):
                layer.inferring = True

    def inference(self,data=None):
        if data is None:
            raise ValueError("inference dataset can't be None.")
        
        predict = self.model(data,training=False)
        return predict

    def save_model(self,save_path):
        pass

    def save_weights(self,save_path):
        self.model.save_weights(save_path)

def bfs_build(params,index,nodes,tensors,results):
        if index >= len(nodes):
            return
        node = nodes[str(index)]
        node_inputs = node["input"]
        for node_input in node_inputs:
            if node_input not in tensors.keys():
                bfs_build(params,node_input,nodes,tensors,results)
        if len(tensors) > len(nodes):
            return
        if node["op"] == "Identity":
            op = L.QIdentity(params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
            results.append(tensor)
        elif node['op'] == "Placeholder":
            op = L.QPlaceholder(params[0],params[1],params[2])
            tensor = op(tensors[-1])
        elif node["op"] == "Conv2D":
            op =  L.QConv2D(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "BatchNorm":
            op =  L.QBatchNorm(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "Relu":
            op =  L.QRelu(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "LeakyRelu":
            op =  L.QLeaky(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "MaxPool":
            op =  L.QMaxPool(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "Add":
            op =  L.QAdd(params[0],params[1],params[2])
            tensor = op([tensors[node["input"][0]],tensors[node["input"][1]]])
        elif node["op"] == "Concat":
            op =  L.QCat(node,params[0],params[1],params[2])
            tensor = op([tensors[node["input"][0]],tensors[node["input"][1]]])
        elif node["op"] == "UpSample":
            op =  L.QUpSample(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "Mean":
            op = L.QAvgPool(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])
        elif node["op"] == "MatMul":
            op = L.QDense(node,params[0],params[1],params[2])
            tensor = op(tensors[node["input"][0]])

        if index not in tensors.keys():
            tensors[index] = tensor
        bfs_build(params,node["output"][0],nodes,tensors,results)

def freeze(layer,qi_list=None,strategy="minmax",num_bits=8,signed=False):
    layer.qo.freeze()
    qw = None
    if hasattr(layer,"kernel"):
        qw = QParam(strategy,num_bits,signed)
        qw.update(layer.kernel,1.0)
        qw.freeze()
        layer.kernel = qw.quantize_tensor(layer.kernel) - qw.zero_point
    if hasattr(layer,"bias"):
        if layer.use_bias:
            layer.bias = quantize_bias(layer.bias,qi_list[0].scale*qw.scale,
                                        zero_point=0,num_bits=32,signed=True)
    if qi_list is not None:
        layer.M0 = []
        layer.n = []
    else:
        return
    for qi in qi_list:
        if not qi.scale:
            qi.freeze()
        if qw is not None:
            M = qi.scale * qw.scale / layer.qo.scale
        else:
            M = qi.scale / layer.qo.scale
        M0,n = searchM0(M,20437)
        layer.M0.append(M0)
        layer.n.append(n)

def freeze_bn(layer,qi,strategy="minmax",num_bits=8,signed=False):
    layer.qo.freeze()
    mean = layer.moving_mean
    var = layer.moving_variance
    std = tf.sqrt(var + layer.epsilon)
    gamma = layer.gamma / std
    beta = layer.beta - layer.gamma*mean / std
    if strategy == "kl":
        strategy = "minmax"
    qw = QParam(strategy,num_bits,signed)
    qw.update(gamma,1.0)
    qw.freeze()
    gamma = qw.quantize_tensor(gamma)
    layer.gamma = tf.reshape(gamma,(1,1,1,-1)) - qw.zero_point
    layer.beta = quantize_bias(beta,scale=qi.scale*qw.scale,
                                    zero_point=0,num_bits=32,signed=True)
    M = qi.scale * qw.scale / layer.qo.scale
    layer.M0 = []
    layer.n = []
    M0,n = searchM0(M,20437)
    layer.M0.append(M0)
    layer.n.append(n)

def freeze_leaky(layer,qi,strategy="minmax",num_bits=8,signed=False):
    layer.qo.freeze()
    layer.weight = tf.constant([1,layer.alpha])
    if strategy == "kl":
        strategy = "minmax"
    qw = QParam(strategy,num_bits,signed)
    qw.update(layer.weight,1.0)
    qw.freeze()
    layer.weight = qw.quantize_tensor(layer.weight) - qw.zero_point
    M = qi.scale * qw.scale / layer.qo.scale
    layer.M0 = []
    layer.n = []
    M0,n = searchM0(M,20437)
    layer.M0.append(M0)
    layer.n.append(n)
