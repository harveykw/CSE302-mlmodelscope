from ..onnxruntime_abc import ONNXRuntimeAbstractClass

import warnings 

import numpy as np
import cv2 

class ONNXRuntime_MLCommons_SSD_ResNet_34_1200x1200(ONNXRuntimeAbstractClass):
  warnings.warn("The batch size should be 1.") 
  def __init__(self, providers):
    model_file_url = "https://s3.amazonaws.com/store.carml.org/models/onnxruntime/resnet34-ssd1200.onnx" 
    model_path = self.model_file_download(model_file_url) 

    # Because this model has only one input, predict method will be replaced with predict_onnx method 
    self.load_onnx(model_path, providers) 

  def maybe_resize(self, img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
      im_height, im_width, _ = dims
      img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img
  
  def pre_process_coco_resnet34(self, img, dims=None, need_transpose=False):
    img = self.maybe_resize(img, dims) 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img / 255. - mean
    img = img / std
    if need_transpose:
      img = img.transpose([2, 0, 1])
    return img 

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.pre_process_coco_resnet34(cv2.imread(input_images[i]), [1200, 1200, 3], True) 
    model_input = np.asarray(input_images) 
    return model_input

  def postprocess(self, model_output):
    n = len(model_output[0])
    probabilities = []
    classes = []
    boxes = []
    for i in range(n):
      probabilities.append([])
      classes.append([])
      boxes.append([])
      detection_boxes = model_output[0][i]
      detection_classes = model_output[1][i]
      scores = model_output[2][i]
      for detection in range(len(scores)):
        if scores[detection] < 0.5:
          break
        probabilities[-1].append(scores[detection])
        classes[-1].append(float(detection_classes[detection]))
        box = detection_boxes[detection]
        boxes[-1].append([box[1], box[0], box[3], box[2]])
    return probabilities, classes, boxes 
