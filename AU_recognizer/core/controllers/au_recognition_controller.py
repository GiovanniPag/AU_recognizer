from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.user_interface.dialogs.dialog import SelectMeshDialog
from AU_recognizer.core.user_interface.views import AURecognitionView
from AU_recognizer.core.util import logger


class AURecognitionController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None
        self.scanning = False

    def bind(self, v: AURecognitionView):
        logger.debug(f"bind in AURecognition controller")
        self.view = v
        self.view.create_view()
        self.view._au_button.configure(command=lambda: self.open_mesh_select_dialog())

    def update_selected(self, data):
        logger.debug(f"update selected in AURecognition controller")
        if not self.data or (self.data and self.data != data):
            self.data = data
            self.view.update_selected_project(data)

    def open_mesh_select_dialog(self):
        logger.debug("open select images for fit dialog")
        if self.data:
            SelectMeshDialog(master=self.master, project=self.data).show()
