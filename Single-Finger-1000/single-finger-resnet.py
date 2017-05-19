
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential,Model
from imagenet_utils import preprocess_input, decode_predictions
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from PIL import Image
import splitImages as split
import numpy as np
import os
import random   

def printDistinguish(text):
    print "\n**--**"
    print str(text)
    print "**--**\n"


def get_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def model():
	model = ResNet50(weights='imagenet')
	model.layers.pop()
	for layer in model.layers:
		layer.trainable=False
	new_layer1 = Dropout(0.4)(model.layers[-1].output)
	new_layer2 = Dense(3,activation="softmax")(new_layer1)
	model_1 = Model(input=model.input, output=[new_layer2])
	model_1.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
	return model_1
	# model_1.summary()

if __name__ == '__main__':
	model_1 = model()
	train_data, test_data, train_labels, test_labels, class_weight = split.split_train_test_data(0.2, 'training-data', 'testPath')
	for image in train_data:
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)
	for image in test_data:
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

	model_1.fit(np.array(train_data),
    	np.array(train_labels),
    		epochs=1,
    			verbose=1,
    				shuffle=True,
    					batch_size=64,
    				        validation_data=(np.array(test_data), np.array(test_labels)),
    					       class_weight=class_weight)
	scores = model_1.evaluate(np.array(test_data), np.array(test_labels), verbose=0)

	print (str(model_1.metrics_names) + "\n" + str(scores) + "\n")

	predicted_labels = model_1.predict(np.array(test_data))
	
	predicted_labels = np.argmax(predicted_labels, axis=1)
	
	target_names = ['Fut', 'Lum', 'Sec']

	print predicted_labels

	conf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), predicted_labels)

	printDistinguish("CONFUSION MATRIX:\n"+ str(conf_matrix) )

	for i in range(3):
		print "Precision for class " + target_names[i]
		sum = 0.0
		for j in range(3):
			sum = sum + conf_matrix[i][j]
		precision_class = conf_matrix[i][i] / sum
		print ": " + str(precision_class) 

	test_labels_generated = []
	for list in test_labels:
		if list[0] == 1:
			test_labels_generated.append(0)
		elif list[1] == 1:
			test_labels_generated.append(1)
		else:
			test_labels_generated.append(2)

	test_path = "testPath"
	listings = os.listdir(test_path)
	f = open('image.html','w')
	message="""<html><body>"""
	for file in listings:
		actual = ""
		predict = ""
		path= test_path + "/" + file
		file=int(file)
		if(test_labels_generated[file]== 0):
			actual = "fut"
		elif (test_labels_generated[file]== 1):
			actual = "lum"
		elif (test_labels_generated[file]== 2):
			actual = "sec"

		if(predicted_labels[file]== 0):
			predict = "fut"
		elif(predicted_labels[file]== 1):
			predict = "lum"
		elif(predicted_labels[file]== 2):
			predict = "sec"
		if predict != actual:
			message += """<figure>
			<p style="float: left; font-size: 9pt; text-align: center; width: 18%; margin-right: 2%; margin-bottom: 0.5em;">
			<img src=""" + path  + " " + """alt="Mountain View" style="width: 100%">
			<caption> true class = """ + actual + "," + " " + """predicted class=""" + predict +  """</caption></p>
			</figure>"""
	message += """</body></html>"""
	f.write(message)
	f.close()