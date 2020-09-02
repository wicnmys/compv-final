import numpy as np
import keras
from keras.preprocessing import image
from pyquaternion import Quaternion
import os
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, sources,batch_size=32, dim=(32,32,32), n_channels=3,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.sources = sources
        self.indices = list(range(len(sources)))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def get_all_labels(self):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        y = self.__data_label_generation(self.indices[0:self.__len__()*self.batch_size])

        return y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __data_label_generation(self, batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((len(batch), 7))

        # Generate data
        for i, ID in enumerate(batch):
            # Store sample
            source = self.sources[ID]

            # process label
            rotation_matrix = np.load(os.path.join(source,"GT/GT_R12.npy"))
            translation_vector = np.load(os.path.join(source,"GT/GT_t12.npy"))
            rotation_quaternion = Quaternion(matrix=rotation_matrix).elements
            label = np.append(rotation_quaternion, translation_vector)

            # Store class
            y[i,] = label

        return y

    def __data_generation(self, batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 7))
        p1 = np.empty((self.batch_size, 300))
        K1 = np.empty((self.batch_size, 9))
        p2 = np.empty((self.batch_size, 300))
        K2 = np.empty((self.batch_size, 9))

        # Generate data
        for i, ID in enumerate(batch):
            # Store sample
            source = self.sources[ID]

            # process images
            img1 = image.load_img(os.path.join(source,"inputs","im1.jpg"), target_size=self.dim)
            img1 = image.img_to_array(img1)
            img2 = image.load_img(os.path.join(source,"inputs","im2.jpg"), target_size=self.dim)
            img2 = image.img_to_array(img2)
            img1 = img1.astype('float32')
            img1 /= 255
            img2 = img2.astype('float32')
            img2 /= 2255

            # process label
            rotation_matrix = np.load(os.path.join(source,"GT/GT_R12.npy"))
            translation_vector = np.load(os.path.join(source,"GT/GT_t12.npy"))
            rotation_quaternion = Quaternion(matrix=rotation_matrix).elements
            label = np.append(translation_vector,rotation_quaternion)

            point_matches1 = np.load(os.path.join(source,"inputs/points1.npy"))
            point_matches2 = np.load(os.path.join(source,"inputs/points2.npy"))
            k1 = np.load(os.path.join(source,"inputs/K1.npy"))
            k2 = np.load(os.path.join(source, "inputs/K2.npy"))

            p1[i,] = point_matches1[:,random.sample(range(point_matches1.shape[1]),k=100)].flatten()
            p2[i,] = point_matches1[:,random.sample(range(point_matches2.shape[1]),k=100)].flatten()
            K1[i,] = k1.flatten()
            K2[i,] = k2.flatten()
            X1[i,] = img1
            X2[i,] = img2

            # Store class
            y[i,] = label

        return [X1,X2, K1, K1, p1, p2], y