import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_hub as hub


class Tensorflow_ResNet():
	def __init__(self): 
		self.model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/5")])
		self.model.build([None, 224, 224, 3])
		for layer in self.model.layers:
			layer.trainable = False
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	def preprocess(cts, image_path):
		for img in image_path:
			#img = load_img(img, target_size=(299, 299))
			image = img_to_array(img)
			image = tf.image.resize(image, (224,224))
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