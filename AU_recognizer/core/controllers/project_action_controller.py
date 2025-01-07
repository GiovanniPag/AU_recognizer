from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.controllers.model_fit_controller import ModelFitController
from AU_recognizer.core.controllers.au_recognition_controller import AURecognitionController
from AU_recognizer.core.util import logger
from AU_recognizer.core.views import ProjectActionView


class ProjectActionController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self._model_fit_controller = ModelFitController(master)
        self._au_recognition_controller = AURecognitionController(master)

    def bind(self, v: ProjectActionView):
        logger.debug(f"bind in project action controller")
        self.view = v
        self.view.create_view()
        self.view.bind_controllers(model_fit_controller=self._model_fit_controller,
                                   au_recognition_controller=self._au_recognition_controller)

    def select_project(self, data):
        logger.debug(f"update selected project in project action controller")
        self.view.update_selected_project(data)
        self._model_fit_controller.update_selected(data)
        self._au_recognition_controller.update_selected(data)
