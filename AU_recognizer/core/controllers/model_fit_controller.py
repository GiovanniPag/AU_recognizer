from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.util import logger
from AU_recognizer.core.views.dialogs.dialog import SelectFitImageDialog
from AU_recognizer.core.views.view import ModelFitView


class ModelFitController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None
        self.scanning = False

    def bind(self, v: ModelFitView):
        logger.debug(f"bind in ModelFit controller")
        self.view = v
        self.view.create_view()
        self.view.select_images_button.config(command=lambda: self.open_image_select_dialog(data=self.view.get_form()))

    def update_selected(self, data):
        logger.debug(f"update selected in ModelFit controller")
        if not self.data or (self.data and self.data != data):
            self.data = data
            self.view.update_selected_project(data)

    def open_image_select_dialog(self, data=None):
        logger.debug("open select images for fit dialog")
        if data:
            SelectFitImageDialog(master=self.master, data=data, project=self.data).show()
