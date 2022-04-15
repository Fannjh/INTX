# !usr/bin/env python
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/quantization/quantizer.py
# Author: FanJH
# Description: 
#############################################
from intx.quantization.model import TFModel
from intx.quantization.node import extract_tf_nodes

QUANT_STRATEGY = ['minmax','kl']
class Quantizer():
    """This is the class for model quantization.
    """
    def __init__(self,strategy,mode,num_bits=8,signed=False):
        super().__init__()
        self.strategy = strategy
        if self.strategy not in QUANT_STRATEGY:
            raise ValueError("quantization stragety only support:",QUANT_STRATEGY)
        self.mode = mode
        self.num_bits = num_bits
        if self.strategy in ["kl"]:
            self.signed = True
        else:
            self.signed = signed

    def __call__(self,model,model_type,input_layer):
        if model_type == "tf":
            nodes = extract_tf_nodes(model)
            model = TFModel(self.mode,self.strategy,self.num_bits,self.signed)
            model.build(input_layer,nodes)
        elif model_type == "pt":
            pass

        return model
        
    def prepare(self,model):
        pass


