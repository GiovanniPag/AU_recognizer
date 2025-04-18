from configparser import ConfigParser
from tkinter import E, StringVar, NE, NW
from typing import Optional

from AU_recognizer.core.model.model_manager import load_model_class
from AU_recognizer.core.user_interface import CustomButton
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import (nect_config, CONFIG, logger, i18n, MF_SELECT_IMG, MF_MODEL, MODEL)


class ModelFitView(View):
    def __init__(self, master=None, model=nect_config[CONFIG][MODEL]):
        super().__init__(master)
        self.master = master
        self.model = model
        self.columnconfigure(0, weight=1)
        self._project_info: Optional[ConfigParser] = None
        self.model_view = None
        self.select_images_button_label = StringVar()
        self.select_images_button: Optional[CustomButton] = None
        self.update_language()

    def update_model(self, new_model=nect_config[CONFIG][MODEL], force_change=False):
        logger.debug("model changed, update views")
        if self.model != new_model:
            if load_model_class(self.model) != load_model_class(new_model) or force_change:
                self.model = new_model
                self.model_view = load_model_class(self.model).get_ui_for_fit_data(self)
                self.model_view.create_view()
                self.__update_view()

    def update_language(self):
        logger.debug("update language in ModelFit view")
        if self.model_view is not None:
            self.model_view.update_language()
        self.select_images_button_label.set(i18n.project_actions_fit[MF_SELECT_IMG])

    def create_view(self):
        logger.debug("create view in ModelFit view")
        # Dynamic model-specific form
        try:
            model_class = load_model_class(self.model)  # helper that imports dynamically
            self.model_view = model_class.get_ui_for_fit_data(self)
            self.model_view.create_view()
        except Exception as e:
            logger.error(f"Failed to load model form: {e}")
            self.model_view = None
        # Button to select images
        self.select_images_button = CustomButton(self, textvariable=self.select_images_button_label)

    def __update_view(self):
        logger.debug("update view in ModelFit view")
        if self.model_view is not None:
            self.model_view.update_view()
            self.model_view.grid(row=0, column=0, sticky=NW + E, pady=5, padx=5)
        self.select_images_button.grid(row=1, column=0, sticky=NE, pady=5)

    def update_selected_project(self, data: Optional[ConfigParser] = None):
        logger.debug("update selected project in fit view")
        self._project_info = data
        self.__update_view()

    def get_form(self):
        form = {
            MF_MODEL: self.model,
        }
        if self.model_view:
            form.update(self.model_view.get_data())
        return form
