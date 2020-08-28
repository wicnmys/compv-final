import json
import os
import uuid

class Config:
    _data = {
                "source": {
                    "training": [
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAB8ISX6xLJGwCOHlr_LXlGEa/Alcatraz_courtyard.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAApS9XxO38sqxDXnk3Dau9wa/buddah.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAAvXtjkm_ReDsNYnueG67E-a/duomo.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AADKHz5O5pXyAOa3vnQph8Kpa/eglise_int.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AACzzwAYAwqFj1eObdULj_3Na/fine_arts_palace.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AABu3sFU2x_rc5e4Vv6gWi4pa/gbg.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAAuEjGaqnl7rRsAyz6n5Ilba/kings_college_front.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AACiXSnHYBXY4R4JLeGJuMKsa/linkoping_dkyrka.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AABlT0m1h8YB94v0fjNYUkr2a/lund_cath_large.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAApgu_9Wdxmuw1p068meOiaa/plaza_de_armas.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AACd1kRP4WlPJB1zEBpd3ICba/pumpkin.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AADdsCErEFvAlX0IfP_A5b7Ta/san_marco.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AAAhHl-dq8GSBjlMIKovTXZna/Sri_Mariamman.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AACBQhi75ThqmpLl6nPFiNJZa/uwo_large.zip?dl=1",
                        "https://www.dropbox.com/sh/z08f8zfj6frmafh/AACP9SctQACa7MtYJfJOMiuqa/west_side.zip?dl=1"
                    ],
                    "testing": [
                        "https://www.dropbox.com/sh/l34j58wj5xpuq79/AADuFLWG8vSQ0r2ZyJQyHS6oa/TestSets_part1.zip?dl=1",
                        "https://www.dropbox.com/sh/l34j58wj5xpuq79/AACiKnSvkpM8y6dG0KVfWxzla/TestSets_part2.zip?dl=1"
                    ]
                },
                "target": {
                    "training": "data/train",
                    "testing": "data/test"
                },
                "output": "/content/drive/My Drive/compv/"
            }
    _parameters = {
                      "epochs": 10,
                      "validation_split": 0.9,
                      "batch_size": 32,
                      "shuffle": True,
                      "n_channels": 3,
                      "dim": [227, 227],
                      "landmarks": [],
                      "beta": 10,
                      "input_shape": [227,227,3],
                      "checkpoint": {
                          "save_freq": "epoch",
                          "period": 1,
                          "path": "/content/drive/My Drive/compv/checkpoints/",
                          "identifier":  uuid.uuid1().hex,
                          "filename": "cp-{epoch:04d}.ckpt",
                          "save_weights_only" : True,
                          "verbose" : 1
                      }

                  }
    _debug = False
    _train = True
    _test = False

    def __init__(self, path):
        if os.path.isfile(path):
            with open(path) as config_file:
                config = json.load(config_file)
                self._data = self.__extract(self._data,config["data"])
                self._debug = config["debug"]
                self._parameters = self.__extract(self._parameters, config["parameters"])
                self._parameters["input_shape"] = self._parameters["dim"] + [self._parameters["n_channels"]]
                self._train = config["train"]
                self._test = config["test"]

    def get_parameter(self, property_name):
        if property_name not in self._parameters.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._parameters[property_name]

    def get_path(self, property_name):
        if property_name not in self._parameters.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._data[property_name]

    def is_debug(self):
        return self._debug

    def is_train(self):
        return self._train

    def is_test(self):
        return self._test

    def __extract(self,target, source):
        for key in source.keys():
            if isinstance(source[key], dict):
                for subkey in source[key].keys():
                    target[key][subkey] = source[key][subkey]
            else:
                target[key] = source[key]
        return target

    def get_bundle(self, className):
        if className == "generator":
            return 	{'dim': self._parameters["dim"],
			  'batch_size': self._parameters["batch_size"],
			  'n_channels': self._parameters["n_channels"],
			  'shuffle': self._parameters["shuffle"]}

        if className =="checkpoint":
            chkp_param = self._parameters["checkpoint"]
            if not os.path.isdir(chkp_param["path"]):
                folder = os.path.join(os.getcwd(), "checkpoints")
                print("Setting up checkpoint directory in current working directory: " + folder)
                if not os.path.isdir(folder):
                    os.mkdir("checkpoints")
                chkp_param["path"] = folder

            full_path = os.path.join(chkp_param["path"],chkp_param["identifier"])
            if not os.path.isdir(full_path):
                print("Unique folder for checkpoints does not exist yet, creating at " + full_path)
                os.mkdir(full_path)

            return {
                "save_freq": chkp_param["save_freq"],
                "period": chkp_param["period"],
                "filepath": os.path.join(chkp_param["path"],chkp_param["identifier"]) + "/" + chkp_param["filename"],
                "verbose": chkp_param["verbose"],
                "save_weights_only": chkp_param["save_weights_only"],
            }
        else:
            Exception("Bundle does not exist.")