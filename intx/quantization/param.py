# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/quantization/param.py
# Author: FanJH
# Description: 
#############################################
import numpy as np
from intx.quantization.common import update_hist
from intx.quantization.common import calc_liner_sz,calc_kl_sz
from intx.quantization.common import quantize_tensor,dequantize_tensor
  
class QParam:
    def __init__(self,strategy="minmax",num_bits=8,signed=False):
        self.strategy = strategy
        self.signed = signed
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min_val = None
        self.max_val = None
        if self.strategy == "kl":
            self.bins = 512
            self.hist = None
            self.KL_threshold = 1e-2
        
        if self.signed:
            self.qmin = -2**(num_bits - 1) + 1
            self.qmax = 2**(num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**(num_bits) -1
            

    def update(self,tensor,theta=0.99):
        if not hasattr(tensor,"numpy"):
            return
        tensor = tensor.numpy()
        if self.strategy == "minmax":
            if self.max_val is None:
                self.max_val = tensor.max()
            else:
                # self.max_val = theta*self.max_val + (1-theta)*tensor.max()
                self.max_val = self.max_val if self.max_val >= tensor.max() else tensor.max()

            if self.min_val is None:
                self.min_val = tensor.min()
            else:
                # self.min_val = theta*self.min_val + (1-theta)*tensor.min()
                self.min_val = self.min_val if self.min_val <= tensor.min() else tensor.min()
        elif self.strategy == "kl":
            tensor_data = tensor.flatten()
            if self.max_val is None or self.min_val is None:
                self.max_val = tensor_data.max()
                self.min_val = tensor_data.min()
                self.max_T = abs(self.max_val) if abs(self.max_val) >= abs(self.min_val) else abs(self.min_val)
                # if np.abs(self.max_val) >= np.abs(self.min_val):
                #     self.selcet_postive = True
                #     tensor_data = tensor_data[tensor_data >= 0]
                #     self.max_T = abs(self.max_val)
                # else:
                #     self.selcet_postive = False
                #     tensor_data = tensor_data[tensor_data < 0]
                #     tensor_data = np.abs(tensor_data)
                #     self.max_T = abs(self.min_val)
                tensor_data = np.abs(tensor_data)
                self.interval = self.max_T / self.bins
                self.hist,self.bins_edge = np.histogram(tensor_data,bins=self.bins,range=(0,self.max_T))
                self.hist = np.array(self.hist,np.float32)
            else:
                # if self.selcet_postive:
                #     tensor_data = tensor_data[tensor_data >= 0]
                # else:
                #     tensor_data = tensor_data[tensor_data < 0]
                tensor_data = np.abs(tensor_data)
                update_hist(tensor_data,self.hist,self.interval,self.bins,0.99)

    def freeze(self):
        if self.strategy == "minmax":
            if self.max_val * self.min_val > 0. and not self.signed:
                if self.max_val > 0.:
                    self.min_val = 0.
                else:
                    self.max_val = 0.
            self.scale,self.zero_point = calc_liner_sz(self.min_val,self.max_val,self.qmin,self.qmax,self.signed)
        elif self.strategy == "kl":
            self.scale,self.zero_point,self.threshold,min_KL = calc_kl_sz(self.hist,self.interval,self.KL_threshold,self.num_bits)
            print("=====>min KL:%.8f"%(min_KL))
            # self.scale,self.zero_point = calc_liner_sz(self.min_val,self.max_val,self.qmin,self.qmax)

    def quantize_tensor(self,tensor):
        return quantize_tensor(tensor,self.scale,self.zero_point,self.qmin,self.qmax)

    def dequantize_tensor(self,q_x):
        return dequantize_tensor(q_x,self.scale,self.zero_point)
