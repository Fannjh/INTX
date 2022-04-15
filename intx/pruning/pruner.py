# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/pruning/pruner.py
# Author: FanJH
# Description: 
#############################################

from intx.pruning.model import TFModel

from intx.pruning.node import extract_tf_nodes

class Pruner:
    def __init__(self,mode,strategy,prune_percent):
        self.mode = mode
        self.strategy = strategy
        self.prune_percent = prune_percent

    def __call__(self,model,model_type='tf',input_layer=None):
        if model_type == "tf":
            nodes = extract_tf_nodes(model)
            model = TFModel(self.mode,self.strategy,self.prune_percent,nodes,input_layer)
            model.build()
            return model
        elif model_type == "pt":
            pass