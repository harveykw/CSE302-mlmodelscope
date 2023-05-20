import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array



class Tensorflow_ResNet():
	def __init__(self): 
		self.model = ResNet101(weights='imagenet')
		for layer in self.model.layers:
			layer.trainable = False
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	def preprocess(cts, image_path):
		for img in image_path:
			#img = load_img(img, target_size=(224, 224))
			resized_images = tf.image.resize_with_pad(img, 224, 224, antialias=False)
			image = tf.identity(resized_images)
			#image = img_to_array(img)
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