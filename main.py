
# main that starts the GUI, the program has two parts
# 1) load an image and make a fitting with a 3dmm
# 2) take the fitted mesh and recognize activated AU
from AU_recognizer.AURecognizer import AURecognizer

# if __name__ == "__main__":
#
#     root = tk.Tk()
#     root.title("Tkinter OpenGL Example")
#     root.geometry("800x600")
#     app = Viewer3DGl(root, obj_file_path="AU_recognizer/projects/sda231_dsa/output/2023-08-21-17290400/mesh_coarse.obj")
#     app.pack(fill=tk.BOTH, expand=tk.YES)
#     app.animate = 1
#     root.mainloop()

# if __name__ == "__main__":
#     app = MeshViewerApp(obj_file_path="AU_recognizer/projects/training_data/output/EMOCA_v2_lr_mse_20/a000/mesh_coarse.obj")
#     app.mainloop()

if __name__ == "__main__":
    # create main application
    app = AURecognizer()
    # start the mainloop
    app.mainloop()
