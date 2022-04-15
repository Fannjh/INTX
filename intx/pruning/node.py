# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/pruning/node.py
# Author: FanJH
# Description: 
#############################################
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from collections import OrderedDict

TYPE_TO_STRING = ["float16","float32","float64","int32","uint8","uint16","uint32","uint64","int16","int8","string"]
TF_OP_LIST = ["Placeholder","Conv2D","Pad","MaxPool","AvgPool","FusedBatchNormV3","BiasAdd","Relu","LeakyRelu","ConcatV2","AddV2","Split","ResizeBilinear","Identity"]
LAYER_LIST = ["Placeholder","Conv2D","MaxPool","AvgPool","Mean","Relu","LeakyRelu","Concat","Add","UpSample","Split","MatMul","Identity"]

def fold_tf_nodes(node_dict):
    fold_node_dict = OrderedDict()
    fold_dict = {}
    index = 0
    for name,node in node_dict.items():
        if node["op"]=="Placeholder" or (node["op"]=="Identity" and "Identity" not in name):
            if node["op"]=="Placeholder":
                node["index"] = index
                node["input"] = []
                node["output"] = []
                index += 1
            continue
        elif node["op"] in ["Relu","LeakyRelu","BatchNorm","BiasAdd"]:
            if "Conv2D" in fold_dict.keys() or "MatMul" in fold_dict.keys():
                fold_dict[node["op"]] = node
                continue
        else:
            if "Conv2D" in fold_dict.keys():
                main_node = fold_dict["Conv2D"]
                for fold_node in fold_dict.values():
                    if fold_node["op"] == "Pad":
                        main_node["paddings"] = fold_node["paddings"]
                    elif fold_node["op"] == "BiasAdd":
                        main_node["bias"] = fold_node["bias"]
                    elif fold_node["op"] == "Relu":
                        main_node["activation"] = "relu"
                        fold_node["fold"] = True
                    elif fold_node["op"] == "LeakyRelu":
                        main_node["activation"] = "leaky"
                        main_node["alpha"] = fold_node["alpha"]
                        fold_node["fold"] = True
                    elif fold_node["op"] == "BatchNorm":
                        main_node["bn"] = 1
                        main_node["scale"] = fold_node["scale"]
                        main_node["offset"] = fold_node["offset"]
                        main_node["mean"] = fold_node["mean"]
                        main_node["variance"] = fold_node["variance"]
                        main_node["epsilon"] = fold_node["epsilon"]
                    fold_node["index"] = main_node["index"]
                    
                fold_dict.clear()
            elif "MatMul" in fold_dict.keys():
                main_node = fold_dict["MatMul"]
                for fold_node in fold_dict.values():
                    if fold_node["op"] == "BiasAdd":
                        main_node["bias"] = fold_node["bias"]
                    fold_node["index"] = main_node["index"]
                fold_dict.clear()
            
        if node["op"] not in ["Pad"]:
            node["index"] = index
            index += 1 
        input_nodes = node["input"]
        node["input"] = []
        node["output"] = []
        for input_node in input_nodes:
            if input_node in node_dict.keys():
                if node_dict[input_node]["op"] == "Const":
                    continue
                elif node_dict[input_node]["op"]=="Identity" and "Identity" not in input_node:
                    continue
                node["input"].append(node_dict[input_node]["index"])
        if node["op"] in ["Conv2D","Pad","MatMul"]:
            fold_dict[node["op"]] = node
            if node["op"] == "Pad":
                node["index"] = node["input"][0]
        else:
            #clear fold container
            fold_dict.clear()

    for name,node in node_dict.items():
        if node["op"]=="Identity" and "Identity" not in name:
            continue
        if node["op"] not in LAYER_LIST:
            continue
        if "fold" in node.keys():
            continue
        index = node["index"]
        fold_node_dict[str(index)] = node
        if node["op"] == "Identity":
            node["output"].append(index+1)
        if "input" in node.keys():
            for input_index in node["input"]:
                fold_node_dict[str(input_index)]["output"].append(index)

    return fold_node_dict

def parse_tf_nodes(nodes):
    node_dict = OrderedDict()
    for node in nodes.values():
        if node.op == "Const":
            continue
        op_dict = {}
        node_name = node.name.split("/")[-1]
        op_dict["dtype"] = TYPE_TO_STRING[node.attr["T"].type]
        op_dict["op"] = node.op
        if node.op != "Placeholder":
            op_dict["input"] = []
            for node_input in node.input:
                node_input = node_input.split(":")[0]
                op_dict["input"].append(node_input)
                    
        if node.op == "Placeholder":
            shape = node.attr["shape"].shape.dim
            op_dict["input_shape"] = [dim.size for dim in shape]
            node_dict[node.name] = op_dict
        elif node.op == "Identity" and "Identity" not in node_name:
            input_node = nodes[node.input[0]]
            tensor = tf.make_ndarray(input_node.attr['value'].tensor)
            op_dict["tensor"] =  tensor
            node_dict[node.name] = op_dict
        elif node.op == "Conv2D":
            op_dict["data_format"] = node.attr["data_format"].s.decode("utf-8")
            op_dict["dilations"] = node.attr["dilations"].list.i
            op_dict["padding"] = node.attr["padding"].s.decode("utf-8")
            op_dict["strides"] = node.attr["strides"].list.i
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Identity":
                    op_dict["filter"] = node_dict[input_node]["tensor"]
            node_dict[node.name] = op_dict
        elif node.op == "BiasAdd":
            op_dict["data_format"] = node.attr["data_format"].s.decode("utf-8")
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Identity":
                    op_dict["bias"] = node_dict[input_node]["tensor"]
            node_dict[node.name] = op_dict    
        elif node.op == "Relu":
            node_dict[node.name] = op_dict
        elif node.op == "LeakyRelu":
            op_dict["alpha"] = node.attr["alpha"].f
            node_dict[node.name] = op_dict
        elif node.op == "FusedBatchNormV3":
            op_dict["op"] = "BatchNorm"
            op_dict["data_format"] = node.attr["data_format"].s.decode("utf-8")
            op_dict["epsilon"] = node.attr["epsilon"].f
            input_nodes = node.input
            param_nodes = []
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Identity":
                    param_nodes.append(input_node)
            op_dict["scale"] = node_dict[param_nodes[0]]["tensor"]
            op_dict["offset"] = node_dict[param_nodes[1]]["tensor"]
            op_dict["mean"] = node_dict[param_nodes[2]]["tensor"]
            op_dict["variance"] = node_dict[param_nodes[3]]["tensor"]
            node_dict[node.name] = op_dict
        elif node.op == "Pad":
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Const":
                    op_dict["paddings"] = tf.make_ndarray(nodes[input_node].attr['value'].tensor)
            node_dict[node.name] = op_dict
        elif node.op == "MaxPool":
            op_dict["padding"] = node.attr["padding"].s.decode("utf-8")
            op_dict["strides"] = node.attr["strides"].list.i
            op_dict["size"] = node.attr["ksize"].list.i
            node_dict[node.name] = op_dict
        elif node.op == "Mean":
            op_dict["keep_dims"] = node.attr["keep_dims"].b
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Const":
                    op_dict["axis"] = tf.make_ndarray(nodes[input_node].attr['value'].tensor)
            node_dict[node.name] = op_dict
        elif node.op == "AddV2":
            op_dict["op"] = "Add"
            node_dict[node.name] = op_dict
        elif node.op == "ConcatV2":
            op_dict["op"] = "Concat"
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Const":
                    op_dict["axis"] = tf.make_ndarray(nodes[input_node].attr['value'].tensor)
            node_dict[node.name] = op_dict
        elif node.op in ["ResizeNearestNeighbor","ResizeBilinear"]:
            op_dict["op"] = "UpSample"
            input_nodes = node.input
            op_dict["align_corners"] = node.attr["align_corners"].b
            op_dict["half_pixel_centers"] = node.attr["half_pixel_centers"].b
            op_dict["interpolation"] = "bilinear" if node.op=="ResizeBilinear" else "nearest"
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Const":
                    op_dict["size"] = tf.make_ndarray(nodes[input_node].attr['value'].tensor)
            node_dict[node.name] = op_dict
        elif node.op == "Split":
            op_dict["group"] = node.attr["num_split"].i
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Const":
                    op_dict["axis"] = tf.make_ndarray(nodes[input_node].attr['value'].tensor)
            node_dict[node.name] = op_dict
        elif node.op == "MatMul":
            op_dict["transpose_a"] = node.attr["transpose_a"].b
            op_dict["transpose_b"] = node.attr["transpose_b"].b
            input_nodes = node.input
            for input_node in input_nodes:
                input_node = input_node.split(":")[0]
                if nodes[input_node].op == "Identity":
                    op_dict["weight"] = node_dict[input_node]["tensor"]
            node_dict[node.name] = op_dict
        elif node.op == "Identity" and "Identity" in node.name:
            node_dict[node.name] = op_dict
        else:
            assert 0,"{0}{1}{2}".format("OP:",node.op," currently not support.")
    
    return node_dict

def extract_tf_nodes(model):
    model_name = model._name.replace("_","")
    model.trainable = False
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.input_shape, model.input[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph = frozen_func.graph.as_graph_def()

    nodes = OrderedDict()
    for node in graph.node:
        nodes[node.name] = node
    node_dict = parse_tf_nodes(nodes)
    nodes = fold_tf_nodes(node_dict)

    return nodes