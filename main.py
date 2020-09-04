from camerapose import CameraPose

if __name__ == "__main__":
    model = CameraPose("config.json")
    #model.train()
    model.test()