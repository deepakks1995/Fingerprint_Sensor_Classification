import numpy
import os
import os.path
from PIL import Image
from numpy import *

def convert_images():
	img_rows, img_cols = 224, 224
	data_paths = ['Crm', 'Mor', 'Sag']
	data_processed_paths = ['Crm1', 'Mor2', 'Sag3']
	cls_no = len(data_paths)
	for clsi in range(cls_no):
		files = []
		files.append(os.listdir(data_paths[clsi]))
		for file in files[0]:
			img = Image.open(data_paths[clsi] + '/' + file)
			img = img.resize((img_rows,img_cols))
			gray = img.convert('L')
			gray.save('training-data/' + data_processed_paths[clsi] +'/' + file.split(".")[0], format="JPEG")
	return True;

if __name__ == '__main__':
	convert_images()