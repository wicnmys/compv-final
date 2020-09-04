from camerapose import CameraPose

if __name__ == "__main__":
    model = CameraPose("config.json")
    #model.train()
    test = model.test(r"/home/myriam/projects/compv-final/output/f85f48deeeb411eab8f7ccb0da67463d")
    #print(test)