# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/sample/config.py
# Author: FanJH
# Description: 
#############################################
from easydict import EasyDict as edict

__C                     = edict()
#from config import cfg
cfg                     = __C

#YOLO parmas
__C.YOLO                        = edict()
__C.YOLO.CLASSES                = "/home/fangjh/Dataset/COCO/coco.names"
__C.YOLO.ANCHORS                = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_V4             = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_TINY           = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.STRIDES_TINY           = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5

#Train params
__C.TRAIN                       = edict()
__C.TRAIN.ANNOT_PATH            = "/home/fangjh/Dataset/COCO/train2017.txt"
__C.TRAIN.BATCH_SIZE            = 8
__C.TRAIN.INPUT_SIZE            = [416] # [320,352,384,416,448,480,512,544,576,608]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LR_INIT               = 1e-3
__C.TRAIN.LR_END                = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 3
__C.TRAIN.FIRST_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 20

#Test params
__C.TEST                        = edict()
__C.TEST.ANNOT_PATH             = "/home/fangjh/Dataset/COCO/val2017.txt"
__C.TEST.BATCH_SIZE             = 8
__C.TEST.INPUT_SIZE             = 416
__C.TEST.DATA_AUG               = False
__C.TEST.SCORE_THRESHOLD        = 0.25
__C.TEST.IOU_THRESHOLD          = 0.45
