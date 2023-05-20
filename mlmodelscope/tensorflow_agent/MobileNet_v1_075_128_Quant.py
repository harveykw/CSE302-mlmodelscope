import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_hub as hub


class Tensorflow_ResNet():
	def __init__(self): 
		self.model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/quantops/classification/3")])
		self.model.build([None, 128, 128, 3])
		for layer in self.model.layers:
			layer.trainable = False
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	def preprocess(cts, image_path):
		for img in image_path:
			#img = load_img(img, target_size=(299, 299))
			image = img_to_array(img)
			image = tf.image.resize(image, (128,128))
			image = preprocess_input(image)
			image = tf.expand_dims(image, axis=0)
		return image

	def predict(self, model_input):
		return self.model(model_input)

	def postprocess(self, model_output):
	    probabilities = tf.nn.softmax(model_output, axis = 1)
	    return probabilities.numpy().tolist()

def init():
	return Tensorflow_ResNet()