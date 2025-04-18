from configparser import ConfigParser
from tkinter import StringVar
from typing import Optional

from AU_recognizer.core.user_interface import CustomButton
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import logger, i18n, AU_SELECT_MESH, AU_TAG_MESH


class AUTagView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._au_button_label = StringVar()
        self.au_button: Optional[CustomButton] = None
        self._au_tag_label = StringVar()
        self.tag_button: Optional[CustomButton] = None
        self.update_language()

    def create_view(self):
        logger.debug("update view in AURecognition view")
        self.au_button = CustomButton(self, textvariable=self._au_button_label)
        self.au_button.grid(row=0, column=0, sticky="nsew")
        self.tag_button = CustomButton(self, textvariable=self._au_tag_label)
        self.tag_button.grid(row=1, column=0, sticky="nsew")

    def update_language(self):
        logger.debug("update language in AURecognition view")
        self._au_button_label.set(i18n.project_actions_au_rec[AU_SELECT_MESH])
        self._au_tag_label.set(i18n.project_actions_au_rec[AU_TAG_MESH])

    def update_selected_project(self, data=None):
        logger.debug("update selected project in AURecognition view")
        self._project_info = data
