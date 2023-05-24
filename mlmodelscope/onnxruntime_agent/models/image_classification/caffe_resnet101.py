import numpy as np
import onnxruntime as ort
import onnx
from torchvision import transforms

class Caffe_ResNet101():
    def __init__(self):
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        self.session = ort.InferenceSession('./mlmodelscope/onnxruntime_agent/model_files/cafferesnet101-imagenet.onnx', sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model = onnx.load('./mlmodelscope/onnxruntime_agent/model_files/cafferesnet101-imagenet.onnx')

    def preprocess(self, model_input):
        preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[102.9801 / 255, 115.9465 / 255, 122.7717 / 255], std=[1 / 255, 1 / 255, 1 / 255])
        ])
        post_process_output = list()
        for image in model_input:
            img = image.convert('RGB')
            img = preprocessor(img).numpy()
            post_process_output.append(img[np.newaxis, :, :])
        return post_process_output

    def predict(self, images):
        result = list()
        for image in images:
            result.append(self.session.run([self.output_name], {self.input_name: image}))
        return result

    def postprocess(self, model_output):
        post_process_output = []
        for i in model_output:
            post_process_output.append(np.argmax(i))
        return post_process_output


def init():
    return Caffe_ResNet101()