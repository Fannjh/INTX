# INTX Framework

This is a CNN model quantitation and pruning framework，now it supoort 1-32 bit quantitaton, also with the function of model pruning,
it can greatly compress the CNN model.INTX based on TensorFlow2 development, and it is still improving, welcome to use.

## Toturial（TensorFlow2）
### model quantation
```python
from nnq.quantization.quantizer import Quantizer
#...
model = YOLO3()
input_layer = tf.keras.Input(shape=[416,416,3])
model(input_layer)
model.load_weights(FLAGS.checkpoint)
quantizer = Quantizer(strategy="minmax",mode="ptq",num_bits=8,signed=False)
model = quantizer(model=model,model_type="tf",input_layer=input_layer)
# qat(not recommended)
# for epoch in range(FLAGS.epochs):
# 	train_step(model,trainset)
# ptq(recommended)
testset = Dataset('test')
model.calibrate(testset,sample_N=1024)
pred = model.inference(images_data)
```
### model pruning
```python
from nnq.pruning.pruner import Pruner
#...
#1、build pruner model
pruner = Pruner(strategy='gem',mode='pat',prune_percent=0.5)
model = pruner(model,'tf',input_layer)
#2、soft prune
if epoch%FLAGS.prune_epoch == 0 or epoch == total_epochs-1:
    model.soft_prune()
#3、mask grad to zero
grads = tape.gradient(total_loss, model.trainable_variables)
grads = model.mask_grad(grads)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
#4、prune
pmodel = model.prune()
#5、pmodel finetune
```
### result
|Model|Dataset|mAP|
|:--:|:--:|:--:|
|yolov3 origin|VOC2007_test|65.36%|
|yolov3 fgem(0.5)|VOC2007_test|62.74%|
|yolov3 origin|COCO2017_val|57.50%|
|yolov3 minmax(int8)|COCO2017_val|54.73%|
## finished work
|Paper|Link|
|:--:|:--:|
|Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference|https://arxiv.org/abs/1712.05877v1 |
|TensorRT INT8 Quantization|http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf |
|Learning Efficient Convolutional Networks Through Network Slimming|https://arxiv.org/abs/1708.06519v1 |
|Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration|https://arxiv.org/abs/1811.00250|

## TODO
* [ ] Adapt to more operator