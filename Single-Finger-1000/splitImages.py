import numpy as np 
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from PIL import Image
import random
import os

def split_train_test_data(percentage, data_path, test_path):
	load_data = []
	load_data_labels = []
	class_folders = os.listdir(os.getcwd() + "/" + data_path)
	class_weights = {}
	total_samples = 0
	for cls in class_folders:
		label = int(cls[3]) - 1
		listings = os.listdir(os.getcwd() + "/" + data_path + "/" + cls)
		class_weights[label] = len(listings)
		total_samples = total_samples + len(listings)
		one_hot_encoding = np.zeros(3)
		one_hot_encoding[label] = 1
		one_hot_encoding_list = [one_hot_encoding for _ in range(len(listings))]
		i = 0
		for file in listings:
			img = image.load_img(os.getcwd() + "/" + data_path + "/" + cls + "/"  + file, target_size=(224, 224))
			x = image.img_to_array(img)
			load_data.append(x)
			load_data_labels.append(one_hot_encoding_list[i])
			i += 1
	for cls in range(3):
		class_weights[cls] = float(class_weights[cls]) / total_samples
		print class_weights[cls],' No. of samples:',total_samples
	shuff_indices = [_ for _ in range(len(load_data))]
	random.shuffle(shuff_indices)
	train_data = []
	train_data_labels = [] 
	test_data = []
	test_data_labels = []
	count = 0
	for i in shuff_indices:
		if count < percentage*len(load_data):
			test_data.append(load_data[i])
			test_data_labels.append(load_data_labels[i])
		else:
			train_data.append(load_data[i])
			train_data_labels.append(load_data_labels[i])
		count = count + 1
	save_test_data(test_data, data_path, test_path)
	return train_data, test_data, train_data_labels, test_data_labels, class_weights


def save_test_data(test_data, data_path, test_path):
	for i in range(len(test_data)):
		img = Image.fromarray(test_data[i].astype('uint8'))
		img.save(os.getcwd() + "/" + test_path + "/" + str(i), "JPEG")

if __name__=='__main__':
	split_train_test_data(0.2, 'training-data',  'testPath')
	print 'You are running the Wrong file' 
