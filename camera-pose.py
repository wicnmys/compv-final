# camera-pose.py
# Andrew Kramer

# regresses the relative camera pose between two images using the method 
# presented in https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import tensorflow
import gzip
import sys
import pickle
import keras
import math
import os
import json

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from dataloader import DataLoader
from tensorflow.transformations import quaternion_matrix
from scipy.linalg import svd

beta = 10
gamma = 10
epochs = 10


def pflat(a):
	return a/a[-1]


def combined_loss_function(y_true, y_pred, x1, x2, K1, K2):
	# extract t,R errors
	y_diff=K.square(y_pred-y_true)
	translation_error = K.sqrt(y_diff[0]+y_diff[1]+y_diff[2])
	orientation_quaternion_error = K.sqrt(y_diff[3]+y_diff[4]+y_diff[5]+y_diff[6])

	# extract camera matrices P1,P2
	P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
	orientation_matrix = quaternion_matrix(y_pred[3:])
	P2 = np.apend(orientation_matrix, translation_error, axis=1)
	P2_1 = P2[1,:]
	P2_2 = P2[2,:]
	P2_3 = P2[3,:]

	# normalize the given 2D points and cameras
	x1normalized = np.linalg.inv(K1)*x1
	x2normalized = np.linalg.inv(K2)*x2
	P1normalized = np.linalg.inv(K1)*P1
	P2normalized = np.linalg.inv(K2)*P2

	# Triangulate the 3D points using the two camera matrices and the given 2D points
	X = np.zeros([4, x1.len])  # this is the array of 3D points
	for i in range(x1.len):
		Mi = np.zeros([6, 6])
		Mi[:6, :4] = np.append(P1normalized, P2normalized, axis=0)
		Mi[:, 5] = [[-x1normalized[:, i]], [0], [0], [0]]
		Mi[:, 6] = [[0], [0], [0], -x2normalized[:, i]]
		U, s, VT = svd(Mi)
		solution = VT[:, -1]
		X[:, i] = pflat(solution[1:4])

	# sum the reprojection error over all reprojected points relative to given 2D points of camera 2
	reprojection_error = 0
	for j in range(X.size):
		reprojection_error += K.sqrt(K.square(x1[j] - (P2_1 * X[j]) / (P2_3 * X[j])) + K.square(x2(j) - (P2_2 * X[j]) / (P2_3 * X[j])))

	return K.mean(translation_error + (beta * orientation_quaternion_error) + (gamma * reprojection_error))


def combined_loss_fucntion2(y_true, y_pred, x1, x2, K1, K2):
	# extract t,R errors
	y_diff=K.square(y_pred-y_true)
	translation_error = K.sqrt(y_diff[0]+y_diff[1]+y_diff[2])
	orientation_quaternion_error = K.sqrt(y_diff[3]+y_diff[4]+y_diff[5]+y_diff[6])

	# calculate fundamental matrix
	t = y_pred[:3]
	R = quaternion_matrix(y_pred[3:])
	t_skew=[[0, -t[2], t[1]],[t[2], 0, -t[0]],[-t[1], t[0], 0]]
	E = np.matmul(t_skew, R)
	K2_Tinv = np.linalg.inv(np.transpose(K2))
	K1_inv = np.linalg.inv(K1)
	F = np.matmul(K2_Tinv, E, K1_inv)

	# calculate the epipolar constraint error
	constraint_error = np.matmul(np.transpose(x1), F, x2)

	return K.mean(translation_error + (beta * orientation_quaternion_error) + (gamma * constraint_error))


def custom_objective(y_true, y_pred):
	error = K.square(y_pred - y_true)
	transMag = K.sqrt(error[0] + error[1] + error[2])
	orientMag = K.sqrt(error[3] + error[4] + error[5] + error[6])
	return K.mean(transMag + (beta * orientMag))

def dot_product(v1, v2):
	return sum((a*b) for a,b in zip(v1,v2))

def length(v):
	return math.sqrt(dot_product(v,v))

def compute_mean_error(y_true, y_pred):
	trans_error = 0
	orient_error = 0
	for i in range(0,y_true.shape[0]):
		trans_error += math.acos(dot_product(y_true[i,:3], y_pred[i,:3])/
								 (length(y_true[i,:3]) * length(y_pred[i,:3])))
		orient_error += math.acos(dot_product(y_true[i,3:], y_pred[i,3:])/
								  (length(y_true[i,3:]) * length(y_pred[i,3:])))
	mean_trans = trans_error / y_true.shape[0]
	mean_orient = orient_error / y_true.shape[0]
	return mean_trans, mean_orient


def create_conv_branch(input_shape):
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(11,11),
					 strides=4, padding='valid',
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Conv2D(256, kernel_size=(5,5),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=1))
	model.add(Conv2D(384, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(384, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(256, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	# replace with SPP if possible
	model.add(MaxPooling2D(pool_size=(3,3), strides=2))
	return model

if __name__ == "__main__":

	with open('config.json') as config_file:
		config = json.load(config_file)

	debug = config['debug']
	if not isinstance(debug, bool):
		debug = True

	img_rows, img_cols = 227, 227
	category_IDs = []
	model_name = 'huge_model_10epoch.h5'
	model = None
	# load training and testing data:
	loader = DataLoader(category_IDs, img_rows, img_cols,debug)
	train_data, test_data = loader.get_data()
	train_labels, test_labels = loader.get_labels()
	input_shape = loader.get_input_shape()

	# define structure of convolutional branches
	conv_branch = create_conv_branch(input_shape)
	branch_a = Input(shape=input_shape)
	branch_b = Input(shape=input_shape)

	processed_a = conv_branch(branch_a)
	processed_b = conv_branch(branch_b)

	train_a = train_data[:,0]
	train_b = train_data[:,1]
	# compute distance between outputs of the CNN branches
	# not sure if euclidean distance is right here
	# merging or concatenating inputs may be more accurate
	#distance = Lambda(euclidean_distance, 
	#				  output_shape = eucl_dist_output_shape)([processed_a, processed_b])
	regression = keras.layers.concatenate([processed_a, processed_b])
	regression = Flatten()(regression) # may not be necessary
	output = Dense(7, kernel_initializer='normal', name='output')(regression)
	model = Model(inputs=[branch_a, branch_b], outputs=[output])

	loss_fn = tensorflow.keras.losses.mean_squared_error
	model.compile(loss=loss_fn,
				  optimizer=keras.optimizers.Adam(lr=.0001, decay=.00001),
				  metrics=['accuracy'])

	if os.path.isfile(model_name):
		print("model", model_name, "found")
		model.load_weights(model_name)
		print("model loaded from file")
	else:
		
		model.fit([train_data[:,0], train_data[:,1]], train_labels,
				  batch_size=32,
				  epochs = epochs,
				  validation_split=0.1,
				  shuffle=True)
		model.save_weights(model_name)
		print("model saved as file", model_name)

	pred = model.predict([train_data[:,0], train_data[:,1]])
	train_trans, train_orient = compute_mean_error(pred, train_labels)
	pred = model.predict([test_data[:,0], test_data[:,1]])
	test_trans, test_orient = compute_mean_error(pred, test_labels)
	np.savetxt('pred.txt', pred, delimiter=' ')
	np.savetxt('labels.txt', test_labels, delimiter=' ')

	print('* Mean translation error on training set: %0.2f' % (train_trans))
	print('* Mean orientation error on training set: %0.2f' % (train_orient))
	print('*     Mean translation error on test set: %0.2f' % (test_trans))
	print('*     Mean orientation error on test set: %0.2f' % (test_orient))
