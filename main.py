# main that starts the GUI, the program has two parts
# 1) load an image and make a fitting with a 3dmm
# 2) take the fitted mesh and generate difference heatmap with neutral pose
from AU_recognizer.AURecognizer import AURecognizer

if __name__ == "__main__":
    # create main application
    app = AURecognizer()
    # start the mainloop
    app.mainloop()

# TODO: check error on coarse fitting, return nulls and crashes
# TODO: show return message
# TODO: refactor compare code, and add pose and identity normalization identity
# TODO: try other dataset (found only 3DFACS but is all npy files)
# TODO: save displacement in numypy file as 5023 values in file ID_AUN
# TODO: scegli AU e individuare il muscolo che la attiva segnarlo come overlay
#  sulla texture e vedere su che parte della mesh finisce cosi da mappare i punti della mesh che finiscono nel overlay
#  del muscolo
