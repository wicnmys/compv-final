# camera-pose.py
# Andrew Kramer

# loads data and labels for camera pose estimation as presented in
# https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import tensorflow
import keras
import glob, os
from pyquaternion import Quaternion

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing import image
from keras import backend as K

class DataLoader:

	num_images_1 = 49 # number of images in type 1 sets (1-77)
	num_images_2 = 64 # number of images in type 2 sets (82-128)
	input_shape = []
	train_labels = []
	test_labels = []
	train_data = []
	test_data = []
	input_shape = []
	formatting = False
	debug = True




	# accepts the name of a directory and the name of a .npy file as strings
	# loads data from the given .npy if it exists, otherwise loads data from
	# raw images and saves it to a .npy file for future runs
	# returns a numpy array representation of
	# the image set in the given directiory
	def __load_data(self, directory, landmarks, img_rows, img_cols):
		os.chdir(os.path.dirname(os.path.abspath(__file__)))
		print('Change directory to ' + os.getcwd())
		image_set = []
		label_set = []
		swd = os.getcwd() # save current working directory
		os.chdir("data/" + directory)
		print('Change directory to ' + os.getcwd())
		wd = os.getcwd()
		if not landmarks:
			landmarks = [ name for name in os.listdir(wd) if os.path.isdir(os.path.join(wd, name)) ]

		if self.debug:
			landmarks = landmarks[0:min(len(landmarks),9)]

		for landmark in landmarks:
			new_images = []
			new_labels = []
			
			os.chdir(wd + "/%s" % landmark) # switch to directory for image files of each landmark
			print('Change directory to ' + os.getcwd())
			landmark_dir = os.getcwd()
			if os.path.isfile("data%s.npy" % landmark):
				new_images = np.load("data%s.npy" % landmark)
				new_labels = np.load("labels%s.npy" % landmark)
			else:
				img_folders = glob.glob(landmark_dir + "/*")

				if self.debug:
					img_folders = img_folders[0:min(len(img_folders), 9)]

				for img_folder in img_folders:
					file_path = img_folder + "/inputs/"
					img1 = image.load_img(file_path + "im1.jpg", target_size=(img_rows, img_cols))
					img_array1 = image.img_to_array(img1)

					img2 = image.load_img(file_path + "im2.jpg", target_size=(img_rows, img_cols))
					img_array2 = image.img_to_array(img2)

					#if not self.formatting:
					#	img_array1 = img_array1.reshape(img_array1.shape[0], 3, img_rows, img_cols)
					#	img_array2 = img_array2.reshape(img_array2.shape[0], 3, img_rows, img_cols)
					#else:
					#	img_array1 = img_array1.reshape(img_array1.shape[0], img_rows, img_cols,3)
					#	img_array2 = img_array2.reshape(img_array2.shape[0], img_rows, img_cols,3)

					img_array1 = img_array1.astype('float32')
					img_array1 /= 255
					img_array2 = img_array2.astype('float32')
					img_array2 /= 255

					image_tuple = (img_array1, img_array2)

					rotation_matrix = np.load(img_folder + "/GT/GT_R12.npy")
					translation_vector = np.load(img_folder + "/GT/GT_t12.npy")

					rotation_quaternion =  Quaternion(matrix=rotation_matrix).elements

					label = np.append(rotation_quaternion,translation_vector)
					new_images.append(image_tuple)
					new_labels.append(label)
				np.save("data%s.npy" % landmark, new_images);
				np.save("labels%s.npy" % landmark, new_labels);
			
			if not np.array(image_set).size:
				image_set = new_images
				label_set = new_labels
			else:
				image_set= np.concatenate((image_set, new_images), 0)
				label_set = np.concatenate((label_set, new_labels), 0)

		os.chdir(swd) # switch back to previous working directory
		print('Change directory to ' + os.getcwd())

		#image_set = np.array(image_set)
		## preprocess input
		#if K.image_data_format() == 'channels_first':
		#	image_set = image_set.reshape(image_set.shape[0], 3, img_rows, img_cols)
		#	self.input_shape = (3, img_rows, img_cols)
		#else:
		#	image_set = image_set.reshape(image_set.shape[0], img_rows, img_cols, 3)
		#	self.input_shape = (img_rows, img_cols, 3)
		#image_set = image_set.astype('float32')
		#image_set /= 255

		return np.array(image_set), np.array(label_set)

	# accepts an array of categoryIDs as a parameter
	# loads ground relative pose data for images in those categories
	def __load_labels(self, categoryIDs):
		labels = []
		data_files = ["data/train_data_mvs.txt", "data/test_data_mvs.txt"]
		for file in data_files:
			f = open(file)
			for line in f:
				if line[0].isdigit():
					sl = line.split()
					nextTuple = (int(sl[0]), int(sl[1]), int(sl[2]),
								 float(sl[3]), float(sl[4]), float(sl[5]),
								 float(sl[6]), float(sl[7]), float(sl[8]), float(sl[9]))
					if nextTuple[2] in categoryIDs:
						labels.append(nextTuple);
			f.close()
		return np.array(labels)

	# accepts array of labels with image identifiers
	# returns shuffled labels split into training and testing sets
	def __organize_labels(self, labels):
		np.random.shuffle(labels)
		num_labels = np.array(labels).shape[0]
		train_index = int(0.8*num_labels)
		train_labels = labels[:train_index,:]
		test_labels = labels[train_index:,:]
		np.savetxt('index_spp.txt', train_labels, delimiter=' ')
		return train_labels, test_labels

	# accepts arrays of training and testing labels, 
	# array of images, and array of category IDs
	# returns arrays of image tuples representing the training and
	# testing datasets
	def __organize_data(self, train_labels, test_labels, images, categoryIDs):
		train_data = []
		test_data = []
		for label in train_labels:
			if label[2] <= 77: # currently can't category IDs above 77
				mult = int(categoryIDs.index(label[2]))
				img1_index = (mult * self.num_images_1) + int(label[0]) - 1
				img2_index = (mult * self.num_images_1) + int(label[1]) - 1
				image_tuple = (images[img1_index], images[img2_index])
				train_data.append(image_tuple)
		for label in test_labels:
			if label[2] <= 77: # currently can't category IDs above 77
				mult = int(categoryIDs.index(label[2]))
				img1_index = (mult * self.num_images_1) + int(label[0]) - 1
				img2_index = (mult * self.num_images_1) + int(label[1]) - 1
				image_tuple = (images[img1_index], images[img2_index])
				test_data.append(image_tuple)
		return np.array(train_data), np.array(test_data)

	# accepts arrays of training and testing labels
	# returs same arrays with image identifying data removed
	# final arrays have form: [relative translation, relative orientation]
	#						  [[x, y, z] [q1, q2, q3, q4]]
	def __clean_labels(self, train_labels, test_labels):
		return train_labels[:,3:], test_labels[:,3:]

	# returns shuffled training and test data consisting of
	# lists of image pairs with indicies [pair_num, image_num, width, height, depth]
	def get_data(self):
		if self.debug:
			max_train = min(len(self.train_data), 99)
			max_test = min(len(self.test_data), 99)
		return self.train_data[0:max_train], self.test_data[0:max_test]

	# returns shuffled training and test labels with form:
	# [x, y, z, q1, q2, q3, q4]
	def get_labels(self):
		if self.debug:
			max_train = min(len(self.train_labels), 99)
			max_test = min(len(self.test_labels), 99)
		return self.train_labels[0:max_train], self.test_labels[0:max_test]

	def get_input_shape(self):
		return self.input_shape

	def __init__(self, landmarks, img_rows, img_cols, debug):
		# preprocess input
		if K.image_data_format() == 'channels_first':
			self.input_shape = (3, img_rows, img_cols)
		else:
			self.input_shape = (img_rows, img_cols, 3)
			formatting = True

		self.debug = debug

		self.train_data, self.train_labels = self.__load_data("train",landmarks, img_rows, img_cols)
		self.test_data, self.test_labels = self.__load_data("test",landmarks, img_rows, img_cols)

		#labels = self.__load_labels(categoryIDs)
		#self.train_labels, self.test_labels = self.__organize_labels(labels)
		#self.train_data, self.test_data = self.__organize_data(self.train_labels, self.test_labels, images, categoryIDs)
		#self.train_labels, self.test_labels = self.__clean_labels(self.train_labels, self.test_labels)
		
