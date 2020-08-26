# camera-pose.py
# Andrew Kramer

# loads data and labels for camera pose estimation as presented in
# https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import glob, os



from keras import backend as K


class DataLoader:
    input_shape = []
    label_loc_test = []
    label_loc_train = []
    data_loc_test = []
    data_loc_train = []
    formatting = False
    debug = True
    data_load = False

    sources = []
    test_sources = []

    # accepts the name of a directory and the name of a .npy file as strings
    # loads data from the given .npy if it exists, otherwise loads data from
    # raw images and saves it to a .npy file for future runs
    # returns a numpy array representation of
    # the image set in the given directiory
    def __load_data(self, directory, landmarks, img_rows, img_cols):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print('Change directory to ' + os.getcwd())
        sources = []
        swd = os.getcwd()  # save current working directory
        os.chdir("data/" + directory)
        print('Change directory to ' + os.getcwd())
        wd = os.getcwd()
        if not landmarks:
            landmarks = [name for name in os.listdir(wd) if os.path.isdir(os.path.join(wd, name)) and name[0] != '.']

        if self.debug:
            landmarks = landmarks[0:min(len(landmarks), 9)]

        for landmark in landmarks:

            os.chdir(wd + "/%s" % landmark)  # switch to directory for image files of each landmark
            print('Change directory to ' + os.getcwd())
            landmark_dir = os.getcwd()

            img_folders = [os.path.join(landmark_dir, name) for name in os.listdir(landmark_dir) if
                           os.path.isdir(os.path.join(landmark_dir, name)) and name[0] != '.']

            if not sources:
                sources = img_folders
            else:
                sources.append(img_folders)

        os.chdir(swd)  # switc backto previous working directory
        print('Change directory to ' + os.getcwd())

        return np.array(sources)


    # returns shuffled training and test labels with form:
    # [x, y, z, q1, q2, q3, q4]
    def get_sources(self):
        filter = list(range(len(self.sources)))
        if self.debug:
            max = min(len(self.sources), 99)
            np.random.shuffle(filter)
            filter = filter[0:max]
        return self.sources[filter]

    def get_input_shape(self):
        return self.input_shape

    def __init__(self, directory,landmarks, img_rows, img_cols, debug):
        # preprocess input
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, img_rows, img_cols)
        else:
            self.input_shape = (img_rows, img_cols, 3)
            self.formatting = True

        self.debug = debug

        self.sources = self.__load_data(directory, landmarks, img_rows, img_cols)
