from typing import Optional

from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.util import logger
from AU_recognizer.core.views import Viewer3DView


class Viewer3DController(Controller):

    def __init__(self, master=None) -> None:
        super().__init__()
        self.master = master
        self.view = None
        self.data = None

    def bind(self, v: Viewer3DView):
        logger.debug("bind in Viewer3D controller")
        self.view = v
        self.view.create_view()

    def update_view(self, data: Optional[dict] = None):
        logger.debug(f"update view in Selected file controller")
        self.view.update_selected_file(data)
        if data:
            self.data = data
        else:
            self.data = None
