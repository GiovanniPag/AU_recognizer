# main that starts the GUI, the program has two parts
# 1) load an image and make a fitting with a 3dmm
# 2) take the fitted mesh and recognize activated AU
from AU_recognizer.AURecognizer import AURecognizer

if __name__ == "__main__":
    # create main application
    app = AURecognizer()
    # start the mainloop
    app.mainloop()

# TODO: show fitting process, and return message, also check error on coarse fitting, return nulls and crashes
# TODO: model fit view scrambled ui
