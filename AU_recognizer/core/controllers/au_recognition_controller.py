from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.util import logger
from AU_recognizer.core.views.views.view import AURecognitionView


class AURecognitionController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master

    def bind(self, v: AURecognitionView):
        logger.debug(f"bind in AURecognition controller")
        self.view = v
        self.view.create_view()

    def update_selected(self, data):
        logger.debug(f"update selected in AURecognition controller")
        self.view.update_selected_project(data)
