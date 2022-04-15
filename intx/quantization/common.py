# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/intx/quantization/common.py
# Author: FanJH
# Description: 
#############################################
import numpy as np
import math

def calc_liner_sz(min_val,max_val,qmin_val,qmax_val,signed=False):
    if signed:
        if abs(max_val) >= abs(min_val):
            T = abs(max_val)
        else:
            T = abs(min_val)
        scale = float(T / qmax_val)
        return scale,0
    else: 
        diff = max_val - min_val
        if max_val-min_val < 1e-6:
            diff = abs(max_val)
        scale = float(diff / (qmax_val-qmin_val))
        if abs(scale) < 1e-9:
            zero_point = 0
        else:
            zero_point = round(qmin_val - min_val/scale)
        zero_point = min(qmax_val, max(zero_point,qmin_val))
        return scale,zero_point

def calc_zero_sz(min_val,max_val,num_bits):
    if abs(max_val) >= abs(min_val):
        T = abs(max_val)
    else:
        T = abs(min_val)
    threshold = 2**(num_bits-1) - 1
    scale = T / threshold

    return scale,0

def calc_kl(P_hist,Q_hist):
    sum_kl = 0
    for bin_ in range(len(P_hist)):
        if P_hist[bin_]!=0. and Q_hist[bin_]==0.:
            sum_kl += 1
        elif P_hist[bin_]!=0. and Q_hist[bin_]!=0.:
            sum_kl += P_hist[bin_]*np.log(P_hist[bin_]/Q_hist[bin_])

    return sum_kl

def calc_kl_sz(hist,hist_interval,KL_THRESHOLD,num_bits):
    target_bin = 2**(num_bits-1)
    threshold_sum = np.sum(hist[target_bin:])
    min_KL = 1e10
    min_KL_scale = None
    min_KL_zeropoint = 0
    min_KL_threshold = target_bin
    for threshold in range(target_bin,len(hist)):
        P_hist = hist[:threshold].copy()
        P_hist[threshold-1] += threshold_sum
        threshold_sum -= hist[threshold]
        num_per_bin = threshold / target_bin
        Q_hist = [0. for _ in range(target_bin)]
        for bin_ in range(target_bin):
            start = bin_ * num_per_bin
            end = start + num_per_bin
            left_upper = math.ceil(start)
            right_lower = math.floor(end)
            Q_hist[bin_] = np.sum(hist[left_upper:right_lower])
            if left_upper > start:
                Q_hist[bin_] += (left_upper-start)*hist[left_upper-1]
            if right_lower < end:
                Q_hist[bin_] += (end-right_lower)*hist[right_lower]

        Q_expand = [0. for _ in range(threshold)]
        for bin_ in range(target_bin):
            start = bin_ * num_per_bin
            end = start + num_per_bin
            left_upper = math.ceil(start)
            right_lower = math.floor(end)

            count = 0.
            for ind in range(left_upper,right_lower):
                if hist[ind] != 0:
                    count += 1
            if left_upper>start and hist[left_upper-1]!=0:
                count += left_upper-start
            if right_lower<end and hist[right_lower]!=0:
                count += end-right_lower
            
            if count > 0.:
                expand_value = Q_hist[bin_] / count
            else:
                expand_value = 0.
            for ind in range(left_upper,right_lower):
                if hist[ind] != 0.:
                    Q_expand[ind] += expand_value
            if left_upper>start and hist[left_upper-1]!=0.:
                Q_expand[left_upper-1] += (left_upper-start)*expand_value
            if right_lower<end and hist[right_lower]!=0.:
                Q_expand[right_lower] += (end-right_lower)*expand_value

        P_hist = P_hist / (np.sum(P_hist)+1e-10)
        Q_expand = Q_expand / (np.sum(Q_expand)+1e-10)
        KL = calc_kl(P_hist,Q_expand)
        if KL < min_KL:
            min_KL = KL
            T = (threshold+0.5)*hist_interval
            min_KL_scale = T / (target_bin-1)
            min_KL_threshold = threshold
            if min_KL < KL_THRESHOLD:
                break

    return min_KL_scale,min_KL_zeropoint,min_KL_threshold,min_KL

def quantize_bias(x,scale,zero_point,num_bits=32,signed=True):
    if scale is None or zero_point is None:
        return x
    
    if scale == 0.:
        return np.zeros(np.shape(x))
    
    if signed:
        qmin = -2**(num_bits-1) + 1
        qmax = 2**(num_bits-1) - 1
    else:
        qmin = 0
        qmax = 2**num_bits -1

    q_x = x / scale + zero_point
    q_x = np.round(q_x)
    q_x = np.clip(q_x,qmin,qmax)
    q_x = q_x.astype(np.int32)
    return q_x

def quantize_tensor(x,scale,zero_point,qmin=None,qmax=None):
    if scale is None or zero_point is None:
        return x
    
    if scale == 0.:
        return np.zeros(np.shape(x))

    q_x = x / scale + zero_point
    q_x = np.round(q_x)
    q_x = np.clip(q_x,qmin,qmax)
    q_x = q_x.astype(np.float32)
    return q_x

def dequantize_tensor(q_x,scale,zero_point):
    if scale is None or zero_point is None:
        return q_x
    q_x = q_x - zero_point
    x = scale * q_x.numpy().astype(np.float32)
    return  x

def searchM0(M:float, P:int=20437):
    n = 1
    while True:
        M0 = int(round(2**n * M))
        approx_result = (M0 * P) >> n
        result = M * P
        error = approx_result -result

        if math.fabs(error) < 1e-6 or n >= 22:
            return M0, n
        n += 1

def update_hist(data,hist,interval,bins,alpha=1.0):
    if hist is None or interval is None:
        return
    ind_list = np.clip(np.floor(data / interval),0,bins-1).astype(np.int)
    bin_counts = np.bincount(ind_list,minlength=len(hist))
    hist = alpha*hist + (1-alpha)*bin_counts.astype(np.float32)

    return

