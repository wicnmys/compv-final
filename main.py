from camerapose import CameraPose

if __name__ == "__main__":
    model = CameraPose("config.json")
    model.train()
    #test = model.test(r"C:\Users\Tomer\Desktop\Weizmann\Multiple view geometry\Project\data sets\out\cf202298eaab11ea86c40242ac1c0002")
    #print(test)