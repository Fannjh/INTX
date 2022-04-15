# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/pruning/common.py
# Author: FanJH
# Description: 
#############################################
from dis import dis
from math import dist
import numpy as np
import tensorflow as tf
from scipy.spatial import distance

def get_prune_layers(model,strategy='gem'):
    prune_layers = []
    for idx,layer in enumerate(model.layers):
        if isinstance(layer,tf.keras.Sequential):
            if layer.name in model.output_names:
                continue
            if strategy != "bn":
                prune_layers.append(layer)
            else:
                is_bn_exist = False
                for sub_layer in layer.layers:
                    if "batch_normalization" in sub_layer.name:
                        is_bn_exist = True
                        break
                if is_bn_exist:
                    prune_layers.append(layer)
        elif "add" in layer.name:
            input_layers= layer.inbound_nodes[0].inbound_layers
            for input_layer in input_layers:
                if input_layer in prune_layers:
                    prune_layers.remove(input_layer)
        elif "up_sampling" in layer.name:
            input_layer = layer.inbound_nodes[0].inbound_layers
            if input_layer in prune_layers:
                prune_layers.remove(input_layer)
        
    return prune_layers

def get_bn_threshold(prune_layers,prune_percent):
    bn_weights = []
    bn_thresholds = []
    for layer in prune_layers:
        for sub_layer in layer.layers:
            if "batch_normalization" in sub_layer.name:
                weight = sub_layer.gamma.numpy()
                bn_weights.extend(np.abs(weight))
                bn_thresholds.append(np.abs(weight).max())
                break
    sorted_bn_weights = np.sort(bn_weights)
    prune_limit_index = (sorted_bn_weights==np.min(bn_thresholds)).nonzero()[0][0]
    percent_limit = prune_limit_index / len(sorted_bn_weights)
    if percent_limit < prune_percent:
        prune_percent = percent_limit
        print("To avoid prune the whole filter, \
            prune percent would not greater %.4f."%percent_limit)
    print("modl prune percent: %.4f."%prune_percent)
    prune_limit_index = int(prune_percent * len(sorted_bn_weights))
    bn_threshold = sorted_bn_weights[prune_limit_index]
    return bn_threshold

def get_bn_mask(prune_layers,bn_threshold):
    bn_mask = []
    for layer in prune_layers:
        for sub_layer in layer.layers:
            if "batch_normalization" in sub_layer.name:
                mask = np.greater_equal(np.abs(sub_layer.gamma.numpy()),bn_threshold).astype(np.int)
                remain = int(mask.sum())
                assert remain > 0, "error,channels will be all pruned,please use lowwer prune_percent."
                bn_mask.append(mask)
                break
    return bn_mask

def get_layer_mask(prune_layers,prune_mask):
    layer_mask = []
    var_num = 0
    for layer in prune_layers:
        if isinstance(layer,tf.keras.Sequential):
            # var_num += len(layer.trainable_variables)
            for var in layer.trainable_variables:
                    if "beta" in var.name or "bias" in var.name:
                        continue
                    var_num += 1
            layer_mask.append(prune_mask[var_num-1])
    return layer_mask

def prune_nodes0(nodes,model,prune_layers,prune_bn_mask):
    for layer in model.layers:
        input_layers = layer.inbound_nodes[0].inbound_layers
        if not isinstance(input_layers,(list,tuple)):
            input_layers = [input_layers]
        beta_offset=None;input_prune_index=None
        for input_layer in input_layers:
            if "up_sampling" in input_layer.name:
                input_layer = input_layer.inbound_nodes[0].inbound_layers
            if input_layer in prune_layers:
                input_activation = None
                for sub_input_layer in input_layer.layers:
                    if "batch_normalization" in sub_input_layer.name:
                        input_bn = sub_input_layer
                    elif "relu" in sub_input_layer.name or "leaky" in sub_input_layer.name:
                        input_activation = sub_input_layer
                input_mask_index = prune_layers.index(input_layer)
                input_prune_mask = prune_bn_mask[input_mask_index]
                beta_offset = input_bn.beta.numpy() * (1-input_prune_mask)
                if input_activation is not None:
                    beta_offset = input_activation(beta_offset).numpy()
                input_node_index = input_layer.node_index
                input_prune_index = np.argwhere(input_prune_mask)[:,0].tolist()
                nodes[str(input_node_index)]["offset"] =  input_bn.beta.numpy()[input_prune_index]
        
        if isinstance(layer,tf.keras.Sequential):
            bn = None
            for sub_layer in layer.layers:
                if "conv2d" in sub_layer.name:
                    conv = sub_layer
                elif "batch_normalization" in sub_layer.name:
                    bn = sub_layer
            if beta_offset is not None:
                kernel = conv.kernel.numpy().sum(axis=(0,1))
                beta_offset = tf.matmul(beta_offset.reshape(1,-1),kernel)
                beta_offset = beta_offset.numpy().reshape(-1)
            node_index = layer.node_index
            prune_mask_index = [x for x in range(conv.kernel.shape[-1])]
            if layer in prune_layers:
                mask_index = prune_layers.index(layer)
                prune_mask = prune_bn_mask[mask_index]
                prune_mask_index = np.argwhere(prune_mask)[:,0].tolist()
            else:
                if bn:
                    nodes[str(node_index)]["offset"] =  bn.beta.numpy()[prune_mask_index]
            kernel = conv.kernel.numpy()
            if input_prune_index:
                kernel = kernel[:,:,input_prune_index,:]
            nodes[str(node_index)]["filter"] = kernel[:,:,:,prune_mask_index]
            if bn:
                nodes[str(node_index)]["scale"] =  bn.gamma.numpy()[prune_mask_index]
                nodes[str(node_index)]["mean"] =  bn.moving_mean.numpy()[prune_mask_index]
                nodes[str(node_index)]["variance"] = bn.moving_variance.numpy()[prune_mask_index]
            if conv.use_bias:
                if beta_offset is not None:
                    nodes[str(node_index)]["bias"] = (conv.bias.numpy()+beta_offset)[prune_mask_index]
                else:
                    nodes[str(node_index)]["bias"] = conv.bias.numpy()[prune_mask_index]
            else:
                if beta_offset is not None:
                    nodes[str(node_index)]["bias"] = beta_offset[prune_mask_index]
    return nodes

def prune_nodes(nodes,model,prune_layers,prune_masks):
    for layer in model.layers:
        input_layers = layer.inbound_nodes[0].inbound_layers
        if not isinstance(input_layers,(list,tuple)):
            input_layers = [input_layers]
        prune_offset=None;input_prune_index=None
        for input_layer in input_layers:
            for skip_layer_name in ["max_pool","avg_pool","up_sampling"]:
                if skip_layer_name in input_layer.name:
                    input_layer = input_layer.inbound_nodes[0].inbound_layers
                    break
            if input_layer in prune_layers:
                input_bn = None
                input_activation = None
                for sub_input_layer in input_layer.layers:
                    if "conv" in sub_input_layer.name:
                        input_conv = sub_input_layer
                    if "batch_normalization" in sub_input_layer.name:
                        input_bn = sub_input_layer
                    elif "relu" in sub_input_layer.name or "leaky" in sub_input_layer.name:
                        input_activation = sub_input_layer
                input_mask_index = prune_layers.index(input_layer)
                input_prune_mask = prune_masks[input_mask_index]
                if np.all(input_prune_mask):
                    continue
                if input_bn is not None:
                    prune_offset = input_bn.beta.numpy() * (1.-input_prune_mask)
                # elif input_conv.use_bias:
                #     prune_offset = input_conv.bias.numpy() * (1.-input_prune_mask)
                if input_conv.bias is not None:
                    if prune_offset is not None:
                        prune_offset = (input_conv.bias.numpy() + prune_offset)*(1.-input_prune_mask)
                    else:
                        prune_offset = input_conv.bias.numpy() * (1.-input_prune_mask)
                if prune_offset is not None and input_activation is not None:
                    prune_offset = input_activation(prune_offset).numpy()
                input_node_index = input_layer.node_index
                input_prune_index = np.argwhere(input_prune_mask)[:,0].tolist()
                if input_bn is not None:
                    nodes[str(input_node_index)]["offset"] =  input_bn.beta.numpy()[input_prune_index]
                # if hasattr(input_conv,"offset"):
                #     nodes[str(input_node_index)]["bias"] = input_conv.offset[input_prune_index]
                if input_conv.bias is not None:
                    nodes[str(input_node_index)]["bias"] = input_conv.bias.numpy()[input_prune_index]
        
        if isinstance(layer,tf.keras.Sequential):
            bn = None
            for sub_layer in layer.layers:
                if "conv2d" in sub_layer.name:
                    conv = sub_layer
                elif "batch_normalization" in sub_layer.name:
                    bn = sub_layer
            if prune_offset is not None:
                kernel = conv.kernel.numpy().sum(axis=(0,1))
                prune_offset = tf.matmul(prune_offset.reshape(1,-1),kernel)
                prune_offset = prune_offset.numpy().reshape(-1)
            node_index = layer.node_index
            prune_mask_index = [x for x in range(conv.kernel.shape[-1])]
            if layer in prune_layers:
                mask_index = prune_layers.index(layer)
                prune_mask = prune_masks[mask_index]
                prune_mask_index = np.argwhere(prune_mask)[:,0].tolist()
            # else:
            #     if bn:
            #         nodes[str(node_index)]["offset"] =  bn.beta.numpy()[prune_mask_index]
            kernel = conv.kernel.numpy()
            if input_prune_index:
                kernel = kernel[:,:,input_prune_index,:]
            nodes[str(node_index)]["filter"] = kernel[:,:,:,prune_mask_index]
            if bn:
                nodes[str(node_index)]["scale"] =  bn.gamma.numpy()[prune_mask_index]
                nodes[str(node_index)]["mean"] =  bn.moving_mean.numpy()[prune_mask_index]
                nodes[str(node_index)]["variance"] = bn.moving_variance.numpy()[prune_mask_index]
                if layer not in prune_layers or np.all(prune_mask):
                    nodes[str(node_index)]["offset"] =  bn.beta.numpy()[prune_mask_index]
            # if conv.use_bias:
            #     if prune_offset is not None:
            #         conv.offset = (conv.bias.numpy()+prune_offset)
            #     else:
            #         conv.offset = conv.bias.numpy()[prune_mask_index]
            # else:
            #     if prune_offset is not None:
            #         conv.offset = prune_offset
            # if layer not in prune_layers:
            #     if hasattr(layer,"offset"):
            #         nodes[str(node_index)]["bias"] = conv.offset[prune_mask_index]
            if conv.bias is not None:
                if prune_offset is not None:
                    conv.bias.assign(conv.bias.numpy()+prune_offset)
            else:
                if prune_offset is not None:
                    conv.bias = tf.convert_to_tensor(prune_offset)
            if layer not in prune_layers or np.all(prune_mask):
                if conv.bias is not None:
                    nodes[str(node_index)]["bias"] = conv.bias.numpy()[prune_mask_index]
            
    return nodes

def build_model(nodes,input_layer):
    temp_outputs = {}
    identity_results = []
    layer_num = len(nodes)
    layer_count = 0
    node = nodes['0']
    while(layer_count < layer_num):
        if node['op'] == "Placeholder":
            input_tensor = input_layer
        elif node["op"] == "Conv2D":
            module = tf.keras.Sequential()
            if "paddings" in node.keys():
                pad = tf.keras.layers.ZeroPadding2D(padding=node["paddings"][1:3])
                module.add(pad)
            kernel_initializer = tf.keras.initializers.Constant(node["filter"])
            if "bias" in node.keys():
                bias_initialzier = tf.keras.initializers.Constant(node["bias"])
            else:
                bias_initialzier = None
            conv = tf.keras.layers.Conv2D(filters=node["filter"].shape[-1],
                                            kernel_size=node["filter"].shape[:2],
                                            strides=node["strides"][1:3],
                                            padding=node["padding"],
                                            use_bias= True if "bias" in node.keys() else False,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initialzier
                                            )
            module.add(conv)
            if "bn" in node.keys() and node["bn"] == 1:
                bn = tf.keras.layers.BatchNormalization(epsilon=node["epsilon"], 
                                        beta_initializer=tf.keras.initializers.Constant(node["offset"]), 
                                        gamma_initializer=tf.keras.initializers.Constant(node["scale"]), 
                                        moving_mean_initializer=tf.keras.initializers.Constant(node["mean"]), 
                                        moving_variance_initializer=tf.keras.initializers.Constant(node["variance"])
                                        # gamma_regularizer=tf.keras.regularizers.l1(l1=0.01)
                                        )
                module.add(bn)
            if "activation" in node.keys():
                if node["activation"] == "relu":
                    activation = tf.keras.layers.ReLU()
                elif node["activation"] == "leaky":
                    activation = tf.keras.layers.LeakyReLU(node["alpha"])
                else:
                    raise ValueError("activation '%s' not support."%node["activation"])
                module.add(activation)
            op = module
            op.node_index = node["index"]
            input_index = node["input"][0]
            if str(input_index) in temp_outputs.keys():
                input_tensor = op(temp_outputs[str(input_index)])
            else:
                input_tensor = op(input_tensor)
        elif node["op"] in ["Relu","LeakyRelu"]:
            if node["op"] == "Relu":
                op = tf.keras.layers.ReLU()
            elif node["op"] == "LeakyRelu":
                op = tf.keras.layers.LeakyReLU()
            input_tensor = op(input_tensor)    
        elif node["op"] == "Add":
            op = tf.keras.layers.Add()
            input_indexs = node["input"]
            input_list = []
            for input_index in input_indexs:
                if str(input_index) in temp_outputs.keys():
                    input_list.append(temp_outputs[str(input_index)])
                else:
                    input_list.append(input_tensor)
            input_tensor = op(input_list)
        elif node["op"] == "Concat":
            op = tf.keras.layers.Concatenate(axis=node["axis"])
            input_indexs = node["input"]
            input_list = []
            for input_index in input_indexs:
                if str(input_index) in temp_outputs.keys():
                    input_list.append(temp_outputs[str(input_index)])
                else:
                    input_list.append(input_tensor)
            input_tensor = op(input_list)
        elif node["op"] == "UpSample":
            scale = node["size"] // input_tensor.shape[1:3]
            op = tf.keras.layers.UpSampling2D(size=scale,interpolation=node["interpolation"])
            input_tensor = op(input_tensor)
            # input_tensor = tf.image.resize(input_tensor,size=node["size"],interpolation=node["interpolation"])
        elif node["op"] == "Identity":
            identity_results.append(input_tensor)
        layer_count += 1
        output_indexs = node["output"]
        if len(output_indexs) > 1:
            temp_outputs[str(node["index"])] = input_tensor
        if layer_count < layer_num:
            node = nodes[str(output_indexs[0])]
    return identity_results

def bfs_build_model(index,nodes,tensors,results):
    if index >= len(nodes):
            return
    node = nodes[str(index)]
    node_inputs = node["input"]
    for node_input in node_inputs:
        if node_input not in tensors.keys():
            bfs_build_model(node_input,nodes,tensors,results)
    if len(tensors) > len(nodes):
        return
    if node['op'] == "Placeholder":
        tensor = tensors[-1]
    elif node["op"] == "Conv2D":
        module = tf.keras.Sequential()
        if "paddings" in node.keys():
            pad = tf.keras.layers.ZeroPadding2D(padding=node["paddings"][1:3])
            module.add(pad)
        kernel_initializer = tf.keras.initializers.Constant(node["filter"])
        if "bias" in node.keys():
            bias_initialzier = tf.keras.initializers.Constant(node["bias"])
        else:
            bias_initialzier = None
        conv = tf.keras.layers.Conv2D(filters=node["filter"].shape[-1],
                                        kernel_size=node["filter"].shape[:2],
                                        strides=node["strides"][1:3],
                                        padding=node["padding"],
                                        use_bias= True if "bias" in node.keys() else False,
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initialzier
                                        )
        module.add(conv)
        if "bn" in node.keys() and node["bn"] == 1:
            bn = tf.keras.layers.BatchNormalization(epsilon=node["epsilon"], 
                                    beta_initializer=tf.keras.initializers.Constant(node["offset"]), 
                                    gamma_initializer=tf.keras.initializers.Constant(node["scale"]), 
                                    moving_mean_initializer=tf.keras.initializers.Constant(node["mean"]), 
                                    moving_variance_initializer=tf.keras.initializers.Constant(node["variance"])
                                    # gamma_regularizer=tf.keras.regularizers.l1(l1=0.01)
                                    )
            module.add(bn)
        if "activation" in node.keys():
            if node["activation"] == "relu":
                activation = tf.keras.layers.ReLU()
            elif node["activation"] == "leaky":
                activation = tf.keras.layers.LeakyReLU(node["alpha"])
            else:
                raise ValueError("activation '%s' not support."%node["activation"])
            module.add(activation)
        op = module
        op.node_index = node["index"]
        tensor = op(tensors[node["input"][0]])
    elif node["op"] in ["Relu","LeakyRelu"]:
        if node["op"] == "Relu":
            op = tf.keras.layers.ReLU()
        elif node["op"] == "LeakyRelu":
            op = tf.keras.layers.LeakyReLU()
        tensor = op(tensors[node["input"][0]])
    elif node["op"] == "MaxPool":
        op = tf.keras.layers.MaxPooling2D()
        tensor = op(tensors[node["input"][0]]) 
    elif node["op"] == "Mean":
        if "pool_size" in node.keys():
            op = tf.keras.layers.AveragePooling2D(node["pool_size"],node["strides"],node["padding"])
        else:
            op = tf.keras.layers.GlobalAveragePooling2D()
        tensor = op(tensors[node["input"][0]])   
    elif node["op"] == "Add":
        op = tf.keras.layers.Add()
        tensor = op([tensors[node["input"][0]],tensors[node["input"][1]]])
    elif node["op"] == "Concat":
        op = tf.keras.layers.Concatenate(axis=node["axis"])
        tensor = op([tensors[node["input"][0]],tensors[node["input"][1]]])
    elif node["op"] == "UpSample":
        scale = node["size"] // tensors[node["input"][0]].shape[1:3]
        op = tf.keras.layers.UpSampling2D(size=scale,interpolation=node["interpolation"])
        tensor = op(tensors[node["input"][0]])
        # input_tensor = tf.image.resize(input_tensor,size=node["size"],interpolation=node["interpolation"])
    elif node["op"] == "MatMul":
        if "bias" in node.keys():
            bias_initialzier = tf.keras.initializers.Constant(node["bias"])
        else:
            bias_initialzier = None
        op = tf.keras.layers.Dense(node["weight"].shape[-1],
                                    use_bias=False if bias_initialzier is None else True,
                                    kernel_initializer=tf.keras.initializers.Constant(node["weight"]),
                                    bias_initializer=bias_initialzier,
                                    )
        tensor = op(tensors[node["input"][0]])
    elif node["op"] == "Identity":
        tensor = tensors[node["input"][0]]
        results.append(tensors[node["input"][0]])

    if index not in tensors.keys():
        tensors[index] = tensor
    bfs_build_model(node["output"][0],nodes,tensors,results)

def get_prune_mask(prune_layers,prune_percent,dist_type="l2"):
    norm_prune_rate = prune_percent * 0.9
    dist_prune_rate = prune_percent * 0.1
    norm_masks=[];dist_masks=[]
    for layer in prune_layers:
        for sublayer in layer.layers:
            if "conv" in sublayer.name:
                weight = sublayer.kernel
        # weight = layer.kernel
        norm_prune_num = int(norm_prune_rate*weight.shape[-1])
        dist_prune_num = int(dist_prune_rate*weight.shape[-1])
        weight = tf.reshape(weight,(-1,weight.shape[-1]))
        if dist_type == "l2" or "cosin":
            weight_norm = tf.norm(weight,2,axis=0)
        elif dist_type == "l1":
            weight_norm = tf.norm(weight,1,axis=0)
        mask_index = weight_norm.numpy().argsort()
        norm_mask_index = mask_index[:norm_prune_num]
        norm_mask = np.ones(weight.shape[-1],dtype=np.int)
        norm_mask[norm_mask_index] = 0
        norm_masks.append(norm_mask)

        dist_mask_index = mask_index[norm_prune_num:]
        dist_weight_vec = tf.gather(weight,dist_mask_index,axis=-1).numpy()
        dist_weight_vec = dist_weight_vec.T
        if dist_type == "l2" or "l1":
            similar_matrix = distance.cdist(dist_weight_vec,dist_weight_vec,"euclidean")
        elif dist_type == "cosin":
            similar_matrix = distance.cdist(dist_weight_vec,dist_weight_vec,"consine")
        similar_sum = np.sum(np.abs(similar_matrix),axis=0)
        dist_mask_index = similar_sum.argsort()[:dist_prune_num]
        dist_mask = np.ones(weight.shape[-1],dtype=np.int)
        dist_mask[dist_mask_index] = 0
        dist_masks.append(dist_mask)
        
    return norm_masks,dist_masks 

def get_grad_mask(model,prune_layers):
    grad_mask = []
    for layer in model.layers:
        if layer in prune_layers:
            for sublayer in layer.layers:
                # var_num = len(sublayer.trainable_variables)
                # if var_num > 0:
                #     grad_mask += [1] * var_num
                for var in sublayer.trainable_variables:
                    if "beta" in var.name or "bias" in var.name:
                        grad_mask.append(0)
                    else:
                        grad_mask.append(1)
        elif isinstance(layer,tf.keras.Sequential):
            for sublayer in layer.layers:
                var_num = len(sublayer.trainable_variables)
                if var_num > 0:
                    grad_mask += [0] * var_num
        else:
            var_num = len(layer.trainable_variables)
            if var_num > 0:
                grad_mask += [0] * var_num
    return grad_mask