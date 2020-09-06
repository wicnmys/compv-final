import numpy as np
import os


class SourceLoader:
    ##########################################################################################
    # SOURCE LOADER
    # This class iterates over all folders provided as a data source for either
    # testing or training (defined here as directory) and stored the locations for each
    # data point object ({images, points corr. etc.}) for easy reference in the
    # data generator
    ####################################################################################


    debug = True
    sources = []

    def __load_data(self, directory, landmarks):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print('Change directory to ' + os.getcwd())
        sources = []
        swd = os.getcwd()  # save current working directory
        os.chdir("data/" + directory)
        print('Change directory to ' + os.getcwd())
        wd = os.getcwd()
        if not landmarks:
            landmarks = [name for name in os.listdir(wd) if os.path.isdir(os.path.join(wd, name)) and name[0] != '.']

        for landmark in landmarks:

            os.chdir(wd + "/%s" % landmark)  # switch to directory for image files of each landmark
            print('Change directory to ' + os.getcwd())
            landmark_dir = os.getcwd()

            img_folders = [os.path.join(landmark_dir, name) for name in os.listdir(landmark_dir) if
                           os.path.isdir(os.path.join(landmark_dir, name)) and name[0] != '.']

            if not np.array(sources).size:
                sources = np.array(img_folders)
            else:
                sources = np.concatenate((sources, np.array(img_folders)),0)

        os.chdir(swd)  # switc backto previous working directory
        print('Change directory to ' + os.getcwd())

        return np.array(sources)


    # returns shuffled data sources
    # returns max 99 if debug is true
    def get_sources(self):
        filter = list(range(len(self.sources)))
        if self.debug:
            max = min(len(self.sources), 99)
            np.random.shuffle(filter)
            filter = filter[0:max]
        return self.sources[filter]

    def __init__(self, directory,landmarks,debug):
        self.debug = debug
        self.sources = self.__load_data(directory, landmarks)
