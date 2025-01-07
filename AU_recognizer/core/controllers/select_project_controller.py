from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.util import logger
from AU_recognizer.core.views.view import ProjectInfoView


class SelectedProjectController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master

    def bind(self, v: ProjectInfoView):
        logger.debug(f"bind in Selected project controller")
        self.view = v
        self.view.create_view()

    def update_view(self, data):
        logger.debug(f"update view in Selected project controller")
        self.view.update_selected_project(data)
