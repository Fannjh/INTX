# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/sample/evaluate.py
# Author: FanJH
# Description: 
#############################################
import os
import shutil
from absl import app
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append('..')

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

import utils
from config import cfg
from yolo import YOLOv3, YOLOv4

from dataset import Dataset
from intx.quantization.quantizer import Quantizer
from intx.pruning.pruner import Pruner

flags = tf.compat.v1.flags
flags.DEFINE_string('model', 'yolov3', 'yolo model,yolov3/yolov4')
flags.DEFINE_string('checkpoint', "checkpoint/yolov3_int8.tflite",'checkpoint file') 
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('classes', "/home/fangjh/Dataset/COCO/coco.names", 'classes file')
flags.DEFINE_string('annotation_path', "/home/fangjh/Dataset/COCO/val2017.txt", 'annotation path')
flags.DEFINE_boolean('quantitation', False, 'quant float32 model and inference.')
flags.DEFINE_boolean('save_image', False, 'save drawed image')
FLAGS = flags.FLAGS

def main(_argv):
    input_size = FLAGS.size
    classes = utils.read_class_names(FLAGS.classes)
    num_class = len(classes)
    predicted_dir_path = 'mAP/predicted'
    ground_truth_dir_path = 'mAP/ground-truth'
    save_image_path = "mAP/detection/"
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(save_image_path): shutil.rmtree(save_image_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    if FLAGS.save_image:
        os.mkdir(save_image_path)

    # Build Model
    if FLAGS.checkpoint.split(".")[-1] not in ["h5","onnx"]:
        input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.model == "yolov3":
            conv_tensors = YOLOv3(input_layer,num_class)
        else:
            conv_tensors = YOLOv4(input_layer,num_class)
        model = tf.keras.Model(input_layer, conv_tensors)
        model.summary()
    if FLAGS.checkpoint.endswith(".weights"):
        utils.load_weights(model,FLAGS.checkpoint,FLAGS.model)
    elif os.path.exists(FLAGS.checkpoint+".index"):
        model.load_weights(FLAGS.checkpoint)
    elif FLAGS.checkpoint.endswith(".onnx"):
        import onnxruntime as ort
        sess = ort.InferenceSession(FLAGS.checkpoint)
        input_name = sess.get_inputs()[0].name
    elif FLAGS.checkpoint.endswith(".h5"):
        model = tf.keras.models.load_model(FLAGS.checkpoint)
    elif FLAGS.checkpoint.endswith(".tflite"):
        interpreter = tf.lite.Interpreter(FLAGS.checkpoint)
        interpreter.allocate_tensors()
        print("tflite model loaded.")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        raise ValueError("checkpoint file %s not found."%FLAGS.checkpoint)

    ##Quantized model
    if FLAGS.quantitation:
        quantizer = Quantizer(strategy="minmax",mode="ptq",num_bits=8,signed=False)
        model = quantizer(model=model,model_type="tf",input_layer=input_layer)
        testset = Dataset('test')
        model.calibrate(testset,sample_N=512)

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(FLAGS.annotation_path, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            original_image      = cv2.imread(image_path)
            original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = classes[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            images_data = image_data[np.newaxis, ...].astype(np.float32)
            if FLAGS.checkpoint.endswith(".onnx"):
                conv_tensors = sess.run([],{input_name:images_data})
            elif FLAGS.checkpoint.endswith(".tflite"):
                interpreter.set_tensor(input_details[0]["index"],images_data)
                interpreter.invoke()
                conv_tensors = [interpreter.get_tensor(output_details[idx]["index"]) for idx in range(len(output_details))]
            else:    
                # conv_tensors = model(images_data,training=False)
                conv_tensors = model.inference(images_data)
            pred_bboxes = []
            for scale_id, conv_tensor in enumerate(conv_tensors):
                pred_bbox = utils.decode(conv_tensor,num_class,scale_id,FLAGS.model)
                pred_bbox = tf.reshape(pred_bbox,(images_data.shape[0],-1,num_class+5))
                pred_bboxes.append(pred_bbox)
            pred_bboxes = tf.concat(pred_bboxes,axis=1)
            bboxes = utils.postprocess_boxes(pred_bboxes[0],original_image_size,input_size,cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
            if FLAGS.save_image:
                image = utils.draw_bbox(original_image, bboxes)
                image = Image.fromarray(image)
                image.save(save_image_path + image_name)
            with open(predict_result_path, 'w') as f:
                for idx,bbox in enumerate(bboxes):
                    class_ind = int(bbox[5])
                    if class_ind < 0 or class_ind >= num_class:
                        raise Exception("class index out of classes range,check classes name file.")
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_name = classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
            if num%20 == 0:
                print("%d/%d, image:%s predicted."%(num, num_lines, image_name))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
