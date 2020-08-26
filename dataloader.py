# camera-pose.py
# Andrew Kramer

# loads data and labels for camera pose estimation as presented in
# https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import glob, os
from pyquaternion import Quaternion

from keras.preprocessing import image
from keras import backend as K


class DataLoader:
    input_shape = []
    train_labels = []
    test_labels = []
    train_loc = []
    test_loc = []
    train_data = []
    test_data = []
    input_shape = []
    formatting = False
    debug = True
    data_load = False

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
        loc_set = []
        swd = os.getcwd()  # save current working directory
        os.chdir("data/" + directory)
        print('Change directory to ' + os.getcwd())
        wd = os.getcwd()
        if not landmarks:
            landmarks = [name for name in os.listdir(wd) if os.path.isdir(os.path.join(wd, name)) and name[0] != '.']

        if self.debug:
            landmarks = landmarks[0:min(len(landmarks), 9)]

        for landmark in landmarks:
            new_images = []
            new_labels = []
            new_loc = []

            os.chdir(wd + "/%s" % landmark)  # switch to directory for image files of each landmark
            print('Change directory to ' + os.getcwd())
            landmark_dir = os.getcwd()
            if os.path.isfile("data%s.npy" % landmark) and self.data_load is True:
                new_images = np.load("data%s.npy" % landmark)
                new_labels = np.load("labels%s.npy" % landmark)
            else:
                img_folders = [name for name in os.listdir(landmark_dir) if
                               os.path.isdir(os.path.join(landmark_dir, name)) and name[0] != '.']
                if landmark == "fine_arts_palace":
                    print(img_folders)
                if self.debug:
                    img_folders = img_folders[0:min(len(img_folders), 99)]

                for img_folder in img_folders:
                    file_path = img_folder + "/inputs/"
                    img1 = image.load_img(file_path + "im1.jpg", target_size=(img_rows, img_cols))
                    img_array1 = image.img_to_array(img1)

                    img2 = image.load_img(file_path + "im2.jpg", target_size=(img_rows, img_cols))
                    img_array2 = image.img_to_array(img2)

                    if not self.formatting:
                        img_array1 = img_array1.reshape(3, img_rows, img_cols)
                        img_array2 = img_array2.reshape(3, img_rows, img_cols)
                    else:
                        img_array1 = img_array1.reshape(img_rows, img_cols, 3)
                        img_array2 = img_array2.reshape(img_rows, img_cols, 3)

                    img_array1 = img_array1.astype('float32')
                    img_array1 /= 255
                    img_array2 = img_array2.astype('float32')
                    img_array2 /= 255

                    image_tuple = (img_array1, img_array2)

                    np.save("images.npy")


                    rotation_matrix = np.load(img_folder + "/GT/GT_R12.npy")
                    translation_vector = np.load(img_folder + "/GT/GT_t12.npy")

                    rotation_quaternion = Quaternion(matrix=rotation_matrix).elements

                    label = np.append(rotation_quaternion, translation_vector)
                    new_images.append(image_tuple)
                    new_labels.append(label)
                    new_loc.append(file_path + "images.npy")

                np.save("data%s.npy" % landmark, new_images);
                np.save("labels%s.npy" % landmark, new_labels);

            if not np.array(image_set).size:
                image_set = new_images
                label_set = new_labels
                new_loc = new_loc
            else:
                image_set = np.concatenate((image_set, new_images), 0)
                label_set = np.concatenate((label_set, new_labels), 0)
                loc_set = np.concatenate((loc_set, new_loc),0)

        os.chdir(swd)  # switch back to previous working directory
        print('Change directory to ' + os.getcwd())

        # image_set = np.array(image_set)
        ## preprocess input
        # if K.image_data_format() == 'channels_first':
        #	image_set = image_set.reshape(image_set.shape[0], 3, img_rows, img_cols)
        #	self.input_shape = (3, img_rows, img_cols)
        # else:
        #	image_set = image_set.reshape(image_set.shape[0], img_rows, img_cols, 3)
        #	self.input_shape = (img_rows, img_cols, 3)
        # image_set = image_set.astype('float32')
        # image_set /= 255

        return np.array(image_set), np.array(label_set), np.array(loc_set)

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

    def get_loc(self):
        if self.debug:
            max_train = min(len(self.train_labels), 99)
            max_test = min(len(self.test_labels), 99)

        return self.train_loc[0:max_train], self.test_loc[0:max_test]

    def get_input_shape(self):
        return self.input_shape

    def __init__(self, landmarks, img_rows, img_cols, debug):
        # preprocess input
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, img_rows, img_cols)
        else:
            self.input_shape = (img_rows, img_cols, 3)
            self.formatting = True

        self.debug = debug

        self.train_data, self.train_labels, self.train_loc = self.__load_data("train", landmarks, img_rows, img_cols)
        self.test_data, self.test_labels, self.test_loc = self.__load_data("test", landmarks, img_rows, img_cols)
