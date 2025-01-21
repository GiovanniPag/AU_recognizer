from pathlib import Path
from tkinter import BOTH
from typing import Optional

from AU_recognizer.AURecognizer import AURecognizer
from AU_recognizer.core.user_interface import CustomFrame
from AU_recognizer.core.user_interface.views import CanvasImage, Viewer3DGl
from AU_recognizer.core.user_interface.views.npy_viewer import NpyViewer
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import logger, P_PATH


class Viewer3DView(View):

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._title = None
        self.data = None
        self.master: AURecognizer = master
        self.__canvas_image: Optional[CanvasImage] = None
        self.__canvas_3d: Optional[Viewer3DGl] = None
        self.__npy_v: Optional[NpyViewer] = None
        self.__placeholder = CustomFrame(self)

    def update_language(self):
        logger.debug("update language in Viewer3D view")
        if self.__canvas_3d:
            self.__canvas_3d.update_language()

    def create_view(self):
        logger.debug("create_view in Viewer3D view")
        self.rowconfigure(0, weight=1)  # make grid cell expandable
        self.columnconfigure(0, weight=1)
        self.__placeholder.grid(row=0, column=0, sticky='nswe')
        self.__placeholder.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__placeholder.columnconfigure(0, weight=1)

    def __update_view(self, type_of_file):
        logger.debug("update view in selected Viewer3D view")
        if self.__canvas_3d:
            self.__canvas_3d.display(animate=0)
        for widget in self.__placeholder.winfo_children():
            widget.destroy()
        path = Path(self.data[P_PATH])
        if type_of_file == "image":
            logger.debug("show image in Viewer3D view")
            self.__canvas_image = CanvasImage(placeholder=self.__placeholder, path=path,
                                              can_grab_focus=self.master.can_grab_focus())
            self.__canvas_image.grid(row=0, column=0, sticky='nswe')
        elif type_of_file == "obj":
            logger.debug("show mesh in Viewer3D view")
            # Create a new OpenGL window
            self.__canvas_3d = Viewer3DGl(master=self.master, placeholder=self.__placeholder, obj_file_path=path)
            self.__canvas_3d.create_view()
            self.__canvas_3d.pack(fill=BOTH, expand=True)
            self.__canvas_3d.display(animate=1)
        elif type_of_file == "npy":
            logger.debug("show npy contents in npy_view")
            self.__npy_v = NpyViewer(master=self.__placeholder, path=path)
            self.__npy_v.create_view()
            self.__npy_v.grid(row=0, column=0, sticky='nswe')
        else:
            logger.debug("no file")
            # no file

    def update_selected_file(self, data: Optional[dict] = None):
        logger.debug("update selected file in selected file view")
        if data and data != self.data:
            path = Path(data[P_PATH])
            # is image
            if path.suffix in (".png", ".bmp", ".jpg"):
                self.data = data
                self.__update_view(type_of_file="image")
            # is obj
            if path.suffix == ".obj":
                self.data = data
                self.__update_view(type_of_file="obj")
            # is npy array
            if path.suffix == ".npy":
                self.data = data
                self.__update_view(type_of_file="npy")

    def update_3d(self):
        if self.__canvas_3d:
            self.__canvas_3d.settings_update()
