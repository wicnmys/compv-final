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
from pyquaternion import Quaternion

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras import backend as K

import tensorflow_graphics.geometry.transformation.rotation_matrix_3d as rm3

from sourceloader import SourceLoader
from datagenerator import DataGenerator
from config import Config
from scipy.linalg import svd

class CameraPose():

    _config = []
    _beta = 10
    _gamma = 10

    def __init__(self, path):
        self._config = Config(path)
        self._beta = self._config.get_parameter("beta")

    def _custom_objective(self,y_true, y_pred):
        error = K.square(y_pred[:,0:7] - y_true[:,0:7])
        transMag = K.sqrt(error[0] + error[1] + error[2])
        orientMag = K.sqrt(error[3] + error[4] + error[5] + error[6])
        return K.mean(transMag + (self._beta * orientMag))

    def pflat(self,a):
        return a / a[-1]

    def dot_product(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dot_product(v, v))

    def compute_mean_error(self, y_true, y_pred):
        trans_error = 0
        orient_error = 0
        for i in range(0, y_true.shape[0]):
            trans_error += math.acos(self.dot_product(y_true[i, :3], y_pred[i, :3]) /
                                     (self.length(y_true[i, :3]) * self.length(y_pred[i, :3])))
            orient_error += math.acos(self.dot_product(y_true[i, 3:], y_pred[i, 3:]) /
                                      (self.length(y_true[i, 3:]) * self.length(y_pred[i, 3:])))
        mean_trans = trans_error / y_true.shape[0]
        mean_orient = orient_error / y_true.shape[0]
        return mean_trans, mean_orient

    def _combined_loss_function(self, y_true, y_pred):
        # extract t,R errors
        error = K.square(y_pred[:, 0:7] - y_true[:, 0:7])
        translation_error = K.sqrt(error[0] + error[1] + error[2])
        orientation_quaternion_error = K.sqrt(error[3] + error[4] + error[5] + error[6])

        # extract camera matrices P1,P2
        P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        orientation_matrix = rm3.from_quaternion(y_pred[:,3:7])
        P2 = tensorflow.apend(orientation_matrix, translation_error, axis=1)
        P2_1 = P2[1, :]
        P2_2 = P2[2, :]
        P2_3 = P2[3, :]

        K1 = y_pred[:,7:17]
        K1 = tensorflow.reshape(K1,[-1,3,3])
        K2 = y_pred[:, 17:27]
        K2 = tensorflow.reshape(K2, [-1,3, 3])
        x1 = y_pred[:,27:327]
        x1 = tensorflow.reshape(x1,[-1,3,100])
        x2 = y_pred[:,327:627]
        x2 = tensorflow.reshape(x2,[-1,3,100])

        # normalize the given 2D points and cameras
        x1normalized = tensorflow.linalg.inv(K1) * x1
        x2normalized = tensorflow.linalg.inv(K2) * x2
        P1normalized = tensorflow.linalg.inv(K1) * P1
        P2normalized = tensorflow.linalg.inv(K2) * P2

        # Triangulate the 3D points using the two camera matrices and the givehttps://www.mat.univie.ac.at/~gerald/ftp/book-mfi/mfi1.pdfn 2D points
        X = tensorflow.zeros([4, x1.len])  # this is the array of 3D points
        for i in range(x1.len):
            Mi = np.zeros([6, 6])
            Mi[:6, :4] = np.append(P1normalized, P2normalized, axis=0)
            Mi[:, 5] = [[-x1normalized[:, i]], [0], [0], [0]]
            Mi[:, 6] = [[0], [0], [0], -x2normalized[:, i]]
            U, s, VT = svd(Mi)
            solution = VT[:, -1]
            X[:, i] = self.pflat(solution[1:4])

        # sum the reprojection error over all reprojected points relative to given 2D points of camera 2
        reprojection_error = 0
        for j in range(X.size):
            reprojection_error += K.sqrt(
                K.square(x1[j] - (P2_1 * X[j]) / (P2_3 * X[j])) + K.square(x2[j] - (P2_2 * X[j]) / (P2_3 * X[j])))

        return K.mean(translation_error + (self._beta * orientation_quaternion_error) + (self._gamma * reprojection_error))

    def _combined_loss_fucntion2(self, y_true, y_pred):
        # extract t,R errors
        error = K.square(y_pred[:, 0:7] - y_true[:, 0:7])
        translation_error = K.sqrt(error[0] + error[1] + error[2])
        orientation_quaternion_error = K.sqrt(error[3] + error[4] + error[5] + error[6])


        K1 = y_pred[:,7:16]
        K1 = tensorflow.reshape(K1,[-1,3,3])
        K2 = y_pred[:, 16:25]
        K2 = tensorflow.reshape(K2, [-1,3, 3])
        x1 = y_pred[:,25:325]
        x1 = tensorflow.reshape(x1,[-1,3,100])
        x2 = y_pred[:,325:625]
        x2 = tensorflow.reshape(x2,[-1,3,100])

        # calculate fundamental matrix
        t = y_pred[:,0:3]
        #quaternion = y_pred[:,3:7]
        #tensorflow.map_fn(Quaternion,quaternion)
        #R = tensorflow.map_fn(lambda x: x.rotation_matrix, quaternion)
        R = rm3.from_quaternion(y_pred[:,3:7])
        #R = [x.rotation_matrix for x in quaternion]
        #t_skew = [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]
        e1 = tensorflow.repeat([1., 0, 0], repeats=1, axis=0)
        e2 = tensorflow.repeat([0, 1., 0], repeats=1, axis=0)
        e3 = tensorflow.repeat([0, 0, 1.], repeats=1, axis=0)

        t_skew = tensorflow.concat([tensorflow.map_fn(lambda x: tensorflow.linalg.cross(x,e1), t),
                                   tensorflow.map_fn(lambda x: tensorflow.linalg.cross(x, e2), t),
                                   tensorflow.map_fn(lambda x: tensorflow.linalg.cross(x, e3), t)],axis=1)
        t_skew = tensorflow.reshape(t_skew, [-1,3,3])

        E = tensorflow.linalg.matmul(t_skew, R)
        K2_Tinv = tensorflow.linalg.inv(tensorflow.transpose(K2,perm=[0,2,1]))
        K1_inv = tensorflow.linalg.inv(K1)
        F = tensorflow.matmul(tensorflow.matmul(K2_Tinv, E), K1_inv)


        # calculate the epipolar constraint error
        step = tensorflow.linalg.matmul(tensorflow.transpose(x1, perm=[0,2,1]), F)
        #print(step)
        #print(x2)
        constraint_error = tensorflow.math.reduce_sum(tensorflow.linalg.diag_part(tensorflow.linalg.matmul(step, x2)))

        #print(translation_error)
        #print(orientation_quaternion_error)
        #print(constraint_error)

        #constraint_error = 2

        print("error run")

        return K.mean(translation_error + (self._beta * orientation_quaternion_error) + (self._gamma * constraint_error))

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

    def test(self, path_weights):
        config = self._config
        loader = SourceLoader("test", config.get_parameter("landmarks"), config.is_debug())
        sources = loader.get_sources()
        input_shape = config.input_shape()
        test_generator = DataGenerator(sources, **config.get_bundle("generator"))
        model = self._generate_model(input_shape)

        latest = tensorflow.train.latest_checkpoint(path_weights)
        if latest:
            print("found existing weights, loading...")
            model.load_weights(latest)

        prediction = model.predict(test_generator)
        labels = test_generator.get_all_labels()
        return self.compute_mean_error(labels,prediction)


    def _generate_model(self, input_shape):

        config = self._config

        # define structure of convolutional branches
        conv_branch = self._create_conv_branch(input_shape)
        do_nothing_k = self._create_do_nothing([9])
        do_nothing_p = self._create_do_nothing([300])

        branch_a = Input(shape=config.get_parameter("input_shape"))
        branch_b = Input(shape=config.get_parameter("input_shape"))

        branch_k1 = Input(shape=[9])
        branch_k2 = Input(shape=[9])
        branch_p1 = Input(shape=[300])
        branch_p2 = Input(shape=[300])

        processed_a = conv_branch(branch_a)
        processed_b = conv_branch(branch_b)

        processed_k1 = do_nothing_k(branch_k1)
        processed_k2 = do_nothing_k(branch_k2)
        processed_p1 = do_nothing_p(branch_p1)
        processed_p2 = do_nothing_p(branch_p2)

        # compute distance between outputs of the CNN branches
        # not sure if euclidean distance is right here
        # merging or concatenating inputs may be more accurate
        # distance = Lambda(euclidean_distance,
        #				  output_shape = eucl_dist_output_shape)([processed_a, processed_b])
        regression = keras.layers.concatenate([processed_a, processed_b])

        regression = Flatten()(regression)  # may not be necessary
        siamese = Dense(7, kernel_initializer='normal', name='output')(regression)
        output = keras.layers.concatenate([siamese, processed_k1, processed_k2, processed_p1, processed_p2])

        model = Model(inputs=[branch_a, branch_b, branch_k1, branch_k2, branch_p1, branch_p2], outputs=[output])

        model.compile(loss=self._custom_objective,
                      optimizer=keras.optimizers.Adam(lr=.0001, decay=.00001),
                      metrics=['accuracy'])

        return model


    def train(self):

        # load and save configuration
        config = self._config
        config.save()

        # load training data
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

        model = self._generate_model(input_shape)

        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(**config.get_bundle("checkpoint"))

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

