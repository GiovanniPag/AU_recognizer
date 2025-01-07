from pathlib import Path
from typing import Optional

from AU_recognizer.core.controllers.base_controller import Controller
from AU_recognizer.core.util import logger, open_path_by_os, P_PATH, FV_OPEN_S
from AU_recognizer.core.views.view import SelectedFileView


class SelectedFileController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None

    def bind(self, v: SelectedFileView):
        logger.debug(f"bind in Selected file controller")
        self.view = v
        self.view.create_view()

    def open_file(self, path: Optional[Path] = None):
        logger.debug(f"open file in selected file controller")
        if path:
            logger.debug(f"open file {path}")
            open_path_by_os(path)
        else:
            if self.data:
                logger.debug(f"open last selected file {self.data}")
                path = Path(self.data[P_PATH])
                open_path_by_os(path)
            else:
                logger.warning(f"no file given in open_file")

    def update_view(self, data: Optional[dict] = None):
        logger.debug(f"update view in Selected file controller")
        self.view.update_selected_file(data)
        if data:
            self.data = data
            path = Path(data[P_PATH])
            self.view.set_command(FV_OPEN_S, lambda: self.open_file(path))
        else:
            self.data = None
            self.view.set_command(FV_OPEN_S, ...)
