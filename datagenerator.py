import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, labels, list_IDs ,batch_size=32, dim=(32,32,32), n_channels=1,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.indices = list(range(len(list_IDs)))
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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __data_generation(self, batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, len(self.labels[batch[0]])))

        # Generate data
        for i, ID in enumerate(batch):
            # Store sample
            img = np.load(self.list_IDs[ID])
            X1[i,] = img[0]
            X2[i,] = img[1]
            # Store class
            y[i,] = self.labels[ID]

        return [X1,X2], y