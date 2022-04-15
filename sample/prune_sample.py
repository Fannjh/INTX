# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /INTX_Release/sample/prune_sample.py
# Author: FanJH
# Description: 
#############################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import shutil
import sys
sys.path.append('..')

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import utils
from config import cfg
from yolo import YOLOv3,YOLOv4
from dataset import Dataset
from intx.pruning.pruner import Pruner

flags = tf.compat.v1.flags
flags.DEFINE_string('model', 'yolov3', 'yolov3')
flags.DEFINE_boolean('isfreeze', False,'transfer learning, freeze yolo layer')
flags.DEFINE_boolean('distribute', False,'use multi GPUs training')
flags.DEFINE_integer('prune_epoch',3,'prune model every n epochs')
flags.DEFINE_string('resume','data/yolov3.weights','weights to restore')
flags.DEFINE_string('image', 'data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'output/', 'path to output image')
FLAGS = flags.FLAGS

def demo():
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_class = len(classes)
    input_size = 416
    checkpoint = "data/yolov3.weights"
    image_path   = "data/kite.jpg"

    # Build Model
    if checkpoint.endswith(".weights") or os.path.exists(checkpoint+".index"):
        input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.model == "yolov3":
            conv_tensors = YOLOv3(input_layer,num_class)
        else:
            conv_tensors = YOLOv4(input_layer,num_class)
        model = tf.keras.Model(input_layer, conv_tensors)
        model.summary()
    if checkpoint.endswith(".weights"):
        utils.load_weights(model,checkpoint,FLAGS.model)
    elif os.path.exists(checkpoint+".index"):
        model.load_weights(checkpoint)
    elif checkpoint.endswith(".onnx"):
        import onnx
        import onnxruntime as ort
        sess = ort.InferenceSession(checkpoint)
        input_name = sess.get_inputs()[0].name
    elif checkpoint.endswith(".h5"):
        model = tf.keras.models.load_model(checkpoint)
    else:
        raise ValueError("checkpoint file %s not found."%FLAGS.checkpoint)
    
    # Process demo image
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    #Float32 model image demo
    if checkpoint.endswith(".onnx"):
        conv_tensors = sess.run([],{input_name:np.copy(image_data)})
    else:
        conv_tensors = model(np.copy(image_data),training=False)
    pred_bboxes = []
    for scale_id, conv_tensor in enumerate(conv_tensors):
        pred_bbox = utils.decode(conv_tensor,num_class,scale_id,FLAGS.model)
        pred_bbox = tf.reshape(pred_bbox,(image_data.shape[0],-1,num_class+5))
        pred_bboxes.append(pred_bbox)
    pred_bboxes = tf.concat(pred_bboxes,axis=1)
    bboxes = utils.postprocess_boxes(pred_bboxes[0],original_image_size,input_size,cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

    image = utils.draw_bbox(np.copy(original_image), bboxes)
    image = Image.fromarray(image)
    image.show()
    if os.path.exists(FLAGS.output):
        image.save(FLAGS.output+image_path.split('/')[-1].replace(".jpg","_original.jpg"))
        
    #Pruning model
    input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
    pruner = Pruner(strategy='bn',mode='ptp',prune_percent=0.1)
    model = pruner(model,'tf',input_layer)
    model.soft_prune()
    model = model.prune()
    conv_tensors = model(image_data,training=False)
    pred_bboxes = []
    for scale_id, conv_tensor in enumerate(conv_tensors):
        pred_bbox = utils.decode(conv_tensor,num_class,scale_id,FLAGS.model)
        pred_bbox = tf.reshape(pred_bbox,(image_data.shape[0],-1,num_class+5))
        pred_bboxes.append(pred_bbox)
    pred_bboxes = tf.concat(pred_bboxes,axis=1)
    bboxes = utils.postprocess_boxes(pred_bboxes[0],original_image_size,input_size,cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

    image = utils.draw_bbox(np.copy(original_image), bboxes)
    image = Image.fromarray(image)
    image.show()
    if os.path.exists(FLAGS.output):
        image.save(FLAGS.output+image_path.split('/')[-1].replace(".jpg","_pruned.jpg"))

def create_model(num_class):
    input_layer = tf.keras.layers.Input([416, 416, 3])
    if FLAGS.model == "yolov3":
        conv_tensors = YOLOv3(input_layer,num_class)
    else:
        conv_tensors = YOLOv4(input_layer,num_class)
    model = tf.keras.Model(input_layer, conv_tensors)
    model.summary()
    is_loaded = False
    if FLAGS.resume.endswith(".weights"):
        utils.load_weights(model,FLAGS.resume,FLAGS.model)
        is_loaded = True
    elif os.path.exists(FLAGS.resume+".index"):
        model.load_weights(FLAGS.resume)
        is_loaded = True
    if is_loaded:
        print('Restoring weights from: %s ' % (FLAGS.resume))

    return model

def train():
    trainset = Dataset('train')
    testset = Dataset('test')
    logdir = "data/log/"
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FIRST_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs+second_stage_epochs) * steps_per_epoch
    freeze_layers = utils.load_freeze_layer()
    num_class = len(utils.read_class_names(cfg.YOLO.CLASSES))
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)
    def generate_data_fn(ctx):
        for data in trainset:
            return data
    if FLAGS.distribute:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_model(num_class)
            optimizer = tf.keras.optimizers.Adam()
            dist_trainset = strategy.experimental_distribute_values_from_function(generate_data_fn)
    else:
        model = create_model(num_class)
        optimizer = tf.keras.optimizers.Adam()
    
    input_layer = tf.keras.layers.Input([416, 416, 3])
    #Pruning model
    pruner = Pruner(strategy='gem',mode='pat',prune_percent=0.5)
    model = pruner(model,'tf',input_layer)
    
    def train_step(model,data):
        image_data,target = data
        with tf.GradientTape() as tape:
            conv_tensors = model(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0.
            for scale_id, conv_tensor in enumerate(conv_tensors):
                pred_tensor = utils.decode(conv_tensor,num_class,scale_id,FLAGS.model)
                loss_items = utils.compute_loss(pred_tensor,conv_tensor,*target[scale_id],num_class,scale_id)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            regularization_loss = 0 #[]
            # for variable in model.trainable_variables:
            #     if "gamma" in variable.name:
            #         regularization_loss.append(tf.keras.regularizers.l1(l1=0.01)(variable))
            # regularization_loss = tf.reduce_sum(tf.stack(regularization_loss))
            # total_loss += regularization_loss
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads = model.mask_grad(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if global_steps%20 == 0:
            tf.print("=> EPOCH %2d  GLOBAL STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   regularization loss: %.6f   total_loss: %4.2f" % 
                                                            (global_steps//steps_per_epoch, global_steps,
                                                            total_steps, optimizer.lr.numpy(),
                                                            giou_loss, conf_loss,
                                                            prob_loss, regularization_loss, total_loss))
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5*(cfg.TRAIN.LR_INIT-cfg.TRAIN.LR_END)* \
                                    (1+tf.cos((global_steps-warmup_steps)/(total_steps-warmup_steps)*np.pi))
        optimizer.lr.assign(lr.numpy())
        
        with writer.as_default():
            tf.summary.scalar("lr",optimizer.lr,step=global_steps)
            tf.summary.scalar("train/toal_loss",total_loss,step=global_steps)
            tf.summary.scalar("train/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("train/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("train/prob_loss", prob_loss, step=global_steps)
        writer.flush()

        return total_loss

    def test_step(model,image_data,target):
        conv_tensors = model(image_data,training=False)
        giou_loss=conf_loss=prob_loss=0.
        # optimizing process
        for scale_id, conv_tensor in enumerate(conv_tensors):
            pred_tensor = utils.decode(conv_tensor,num_class,scale_id)
            loss_items = utils.compute_loss(pred_tensor,conv_tensor,*target[scale_id],num_class,scale_id)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        return total_loss

    @tf.function
    def dist_train_step():
        per_replica_losses = strategy.run(train_step,args=(dist_trainset,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,axis=None)
    total_epochs = first_stage_epochs+second_stage_epochs
    for epoch in range(total_epochs):
        if FLAGS.isfreeze:
            if epoch < first_stage_epochs:
                utils.freeze(model,freeze_layers,frozen=True)
            else:
                utils.freeze(model,freeze_layers,frozen=False)
        if FLAGS.distribute:
            for step in range(steps_per_epoch):
                dist_loss = dist_train_step()
                tf.print("==>Dist Train, epoch:%02d, step:%5d, avg train loss:%.2f"%(epoch, step, dist_loss))
        else:
            for step,data in enumerate(trainset):
                train_step(model,data)

        if epoch%FLAGS.prune_epoch == 0 or epoch == total_epochs-1:
            model.soft_prune()

        test_loss = 0.
        for image_data,target in testset:
            test_loss += test_step(model,image_data,target)
        avg_test_loss = test_loss / len(testset)
        tf.print("==>TEST, epoch:%02d, avg test loss:%.2f"%(epoch,avg_test_loss))
        if (epoch+1)%5 == 0:
            model.save_weights("checkpoint/epoch%02d-%.2f"%(epoch+1,avg_test_loss))
    
    pmodel = model.prune()
    pmodel.save("checkpoint/yolov3_gem.onnx",format="onnx")

    global_steps.assign(0)
    for epoch in range(30):
        for step,data in enumerate(trainset):
            train_step(pmodel,data)
            
        test_loss = 0.
        for image_data,target in testset:
            test_loss += test_step(pmodel,image_data,target)
        avg_test_loss = test_loss / len(testset)
        tf.print("==>TEST, epoch:%02d, avg test loss:%.2f"%(epoch,avg_test_loss))
        if (epoch+1)%5 == 0:
            pmodel.save("checkpoint/epoch%02d-%.2f"%(epoch+51,avg_test_loss),"onnx")
    pmodel.save("checkpoint/yolov3_gem_finetune.onnx",format="onnx")

def main():
    demo()
    # train()

if __name__ == "__main__":
    main()
