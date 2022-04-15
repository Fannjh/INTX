# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/pruning/model.py
# Author: FanJH
# Description: 
#############################################
import copy
import tf2onnx
import numpy as np
import tensorflow as tf
import intx.pruning.common as common

class TFModel():
    def __init__(self,mode,strategy,prune_percent,nodes,input_layer):
        super(TFModel,self).__init__()
        self.mode = mode
        self.strategy = strategy
        self.prune_percent = prune_percent
        self.nodes = nodes
        self.input_layer = input_layer
        self.model = None
        self.prune_layers = None
        self.prune_masks = None
        self.grad_masks = None
    
    def build(self,nodes=None):
        if nodes is None:
            nodes = self.nodes
        # identity_results = common.build_model(nodes,self.input_layer)
        tensors = {-1:self.input_layer}
        identity_results = []
        common.bfs_build_model(0,nodes,tensors,identity_results)
        self.model = tf.keras.Model(self.input_layer,identity_results)
        self.model(self.input_layer)
        self.trainable_variables = self.model.trainable_variables

    def __call__(self,input,training=False):
        return self.model(input,training=training)

    def bn_prune(self,prune_percent):
        prune_layers = common.get_prune_layers(self.model)
        bn_threshold = common.get_bn_threshold(prune_layers,prune_percent)
        prune_bn_mask = common.get_bn_mask(prune_layers,bn_threshold)
        nodes = copy.deepcopy(self.nodes)
        nodes = common.prune_nodes(nodes,self.model,prune_layers,prune_bn_mask)
        pruned_model = TFModel(self.mode,self.strategy,nodes)
        return pruned_model

    def prune(self):
        if self.strategy == "bn":
            prune_layers = common.get_prune_layers(self.model,self.strategy)
            bn_threshold = common.get_bn_threshold(prune_layers,self.prune_percent)
            prune_masks = common.get_bn_mask(prune_layers,bn_threshold)
        elif self.strategy == "gem":
            prune_layers = self.prune_layers
            prune_masks = common.get_layer_mask(prune_layers,self.prune_masks)
        nodes = copy.deepcopy(self.nodes)
        nodes = common.prune_nodes(nodes,self.model,prune_layers,prune_masks)
        model = TFModel(self.mode,self.strategy,self.prune_percent,nodes,self.input_layer)
        model.build()
        return model

    def soft_prune(self):
        if self.mode == "ptp":
            return
        if self.prune_layers is None:
            self.prune_layers = common.get_prune_layers(self.model,self.strategy)
        norm_masks,dist_masks = common.get_prune_mask(self.prune_layers,self.prune_percent)
        prune_mask = []
        for idx,layer in enumerate(self.prune_layers):
            bn = None
            for sublayer in layer.layers:
                if "conv" in sublayer.name:
                    conv = sublayer
                elif "batch_normalization" in sublayer.name:
                    bn = sublayer
            mask = np.array(norm_masks[idx],dtype=np.int32).reshape(1,1,1,conv.kernel.shape[-1]) *\
                                        np.array(dist_masks[idx],dtype=np.int32).reshape(1,1,1,conv.kernel.shape[-1])
            prune_mask.append(mask)
            conv.kernel.assign(conv.kernel.numpy() * mask)
            if conv.use_bias:
                mask = np.array(norm_masks[idx],dtype=np.int32) * np.array(dist_masks[idx],dtype=np.int32)
                # prune_mask.append(mask)
                # conv.bias.assign(conv.bias.numpy() * mask)
            if bn:
                mask = np.array(norm_masks[idx],dtype=np.int32) * np.array(dist_masks[idx],dtype=np.int32)
                prune_mask.append(mask)
                bn.gamma.assign(bn.gamma.numpy() * mask)
                # prune_mask.append(mask)
                # bn.beta.assign(bn.beta.numpy() * mask)
        self.prune_masks = prune_mask

    def mask_grad(self,gradients):
        if self.prune_masks is None:
            return gradients
        if self.grad_masks is None:
            self.grad_masks = common.get_grad_mask(self.model,self.prune_layers)
        mask_gradients = []
        prune_ind = 0
        for ind,grad in enumerate(gradients):
            if self.grad_masks[ind]:
                mask_gradients.append(grad * self.prune_masks[prune_ind])
                prune_ind += 1
            else:
                mask_gradients.append(grad)
        return mask_gradients

    def load_weights(self,path):
        self.model.load_weights(path)

    def save_weights(self,save_path):
        self.model.save_weights(save_path)

    def save(self,save_path,format="onnx"):
        if format == "h5":
            save_path = save_path.replace(".h5","")+".h5"
            self.model.save(save_path,save_format='h5')
        elif format == "tf":
            save_path = save_path.replace(".h5","").replace(".onnx","")
            self.model.save(save_path,save_format='tf')
        else:
            save_path = save_path.replace(".onnx","")+".onnx"
            spec = (tf.TensorSpec(self.input_layer.shape,tf.float32,"input"),)
            tf2onnx.convert.from_keras(self.model,spec,opset=13,output_path=save_path)