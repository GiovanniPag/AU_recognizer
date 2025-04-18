from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.user_interface.dialogs.complex_dialog import TagMeshDialog
from AU_recognizer.core.user_interface.views.au_tag_view import AUTagView
from AU_recognizer.core.util import logger


class AUTagController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None
        self.scanning = False

    def bind(self, v: AUTagView):
        logger.debug(f"bind in AURecognition controller")
        self.view = v
        self.view.create_view()
        self.view.tag_button.configure(command=lambda: self.open_mesh_tag_dialog())

    def update_selected(self, data):
        logger.debug(f"update selected in AURecognition controller")
        if not self.data or (self.data and self.data != data):
            self.data = data
            self.view.update_selected_project(data)

    def open_mesh_tag_dialog(self):
        logger.debug("open select images for fit dialog")
        if self.data:
            TagMeshDialog(master=self.master, project=self.data).show()
