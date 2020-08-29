# camera-pose.py
# Andrew Kramer

# regresses the relative camera pose between two images using the method 
# presented in https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import tensorflow
import keras
import math
import os
import json
import re

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras import backend as K

from sourceloader import SourceLoader
from datagenerator import DataGenerator
from config import Config

beta = 10
epochs = 10

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

def checkpoint_path(config):
    path = ""
    if config.checkpoint_path():
        folder = config.checkpoint_path()
        with open(folder) as config_file:
            loc = json.load(config_file)
            if "model_checkpoint_path" in loc.keys():
                path = os.path.join(folder, loc["model_checkpoint_path"])
    return path

if __name__ == "__main__":

    # load and save configuration
    config = Config('config.json')
    config.save()

    # load training and testing data:
    loader = SourceLoader("train", config.get_parameter("landmarks"), config.is_debug())
    sources = loader.get_sources()
    input_shape = config.input_shape()


    len_data = len(sources)
    numbers = list(range(len_data))
    np.random.shuffle(numbers)
    split_data = math.floor(len_data*config.get_parameter('validation_split'))
    train_ids = numbers[0:split_data]
    val_ids = numbers[split_data+1:len(numbers)]
    validation_sources = sources[val_ids]
    training_sources = sources[train_ids]

    training_generator = DataGenerator(training_sources, **config.get_bundle("generator"))
    validation_generator = DataGenerator(validation_sources, **config.get_bundle("generator"))

    # define structure of convolutional branches
    conv_branch = create_conv_branch(input_shape)
    branch_a = Input(shape=config.get_parameter("input_shape"))
    branch_b = Input(shape=config.get_parameter("input_shape"))

    processed_a = conv_branch(branch_a)
    processed_b = conv_branch(branch_b)

    # compute distance between outputs of the CNN branches
    # not sure if euclidean distance is right here
    # merging or concatenating inputs may be more accurate
    #distance = Lambda(euclidean_distance,
    #				  output_shape = eucl_dist_output_shape)([processed_a, processed_b])
    regression = keras.layers.concatenate([processed_a, processed_b])

    regression = Flatten()(regression) # may not be necessary
    output = Dense(7, kernel_initializer='normal', name='output')(regression)
    model = Model(inputs=[branch_a, branch_b], outputs=[output])

    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(**config.get_bundle("checkpoint"))

    model.compile(loss=custom_objective,
                  optimizer=keras.optimizers.Adam(lr=.0001, decay=.00001),
                  metrics=['accuracy'])

    initial_epoch = 0
    latest = tensorflow.train.latest_checkpoint(config.checkpoint_path())
    if latest:
        print("found existing weights, loading...")
        model.load_weights(latest)
        found_num = re.search(r'\d+', os.path.basename(latest))
        if found_num:
            checkpoint_id = int(found_num.group(0))
            initial_epoch = checkpoint_id

        # Re-evaluate the model
        #loss, acc = model.evaluate(validation_generator)
        #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    model.fit(training_generator,
                        validation_data=validation_generator, epochs=config.get_parameter("epochs"),
              initial_epoch=initial_epoch,
              callbacks=[cp_callback])

