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

class CameraPose():

    _config = []
    _beta = 10

    def __init__(self, path):
        self._config = Config(path)
        self._beta = self._config.get_parameter("beta")

    def _custom_objective(self,y_true, y_pred):
        error = K.square(y_pred[:,0:7] - y_true[:,0:7])
        transMag = K.sqrt(error[0] + error[1] + error[2])
        orientMag = K.sqrt(error[3] + error[4] + error[5] + error[6])
        return K.mean(transMag + (self._beta * orientMag))


    def _create_conv_branch(self, input_shape):
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

    def _create_do_nothing(self,input_shape):
        model = Sequential()
        model.add(Lambda(lambda x: x))
        return model

    def _checkpoint_path(self,config):
        path = ""
        if config.checkpoint_path():
            folder = config.checkpoint_path()
            with open(folder) as config_file:
                loc = json.load(config_file)
                if "model_checkpoint_path" in loc.keys():
                    path = os.path.join(folder, loc["model_checkpoint_path"])
        return path

    def run(self):

        # load and save configuration
        config = self._config
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
        conv_branch = self._create_conv_branch(input_shape)
        do_nothing_k = self._create_do_nothing([9])
        do_nothing_p = self._create_do_nothing([300])



        branch_a = Input(shape=config.get_parameter("input_shape"))
        branch_b = Input(shape=config.get_parameter("input_shape"))

        branch_k1  = Input(shape=[9])
        branch_k2 = Input(shape=[9])
        branch_p1 = Input(shape=[300])
        branch_p2 = Input(shape=[300])



        processed_a = conv_branch(branch_a)
        processed_b = conv_branch(branch_b)

        processed_k1  = do_nothing_k(branch_k1)
        processed_k2 = do_nothing_k(branch_k2)
        processed_p1 = do_nothing_p(branch_p1)
        processed_p2 = do_nothing_p(branch_p2)

        # compute distance between outputs of the CNN branches
        # not sure if euclidean distance is right here
        # merging or concatenating inputs may be more accurate
        #distance = Lambda(euclidean_distance,
        #				  output_shape = eucl_dist_output_shape)([processed_a, processed_b])
        regression = keras.layers.concatenate([processed_a, processed_b])

        regression = Flatten()(regression) # may not be necessary
        siamese = Dense(7, kernel_initializer='normal', name='output')(regression)
        output = keras.layers.concatenate([siamese,processed_k1,processed_k2,processed_p1,processed_p2])

        model = Model(inputs=[branch_a, branch_b, branch_k1, branch_k2, branch_p1, branch_p2], outputs=[output])

        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(**config.get_bundle("checkpoint"))

        model.compile(loss=self._custom_objective,
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


        model.fit(training_generator,
                            validation_data=validation_generator, epochs=config.get_parameter("epochs"),
                  initial_epoch=initial_epoch,
                  callbacks=[cp_callback])

