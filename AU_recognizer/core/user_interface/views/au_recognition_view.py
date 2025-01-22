from configparser import ConfigParser
from tkinter import StringVar
from typing import Optional

from AU_recognizer.core.user_interface import CustomLabel
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import logger


class AURecognitionView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._exist_label_info = StringVar()
        self._exist_label: Optional[CustomLabel] = CustomLabel(self, text="exists")
        self.update_language()

    def create_view(self):
        logger.debug("update view in AURecognition view")
        self._exist_label.grid(row=0, column=0, sticky="nsew")

    def update_language(self):
        logger.debug("update language in AURecognition view")

    def update_selected_project(self, data=None):
        logger.debug("update selected project in AURecognition view")
        self._project_info = data
