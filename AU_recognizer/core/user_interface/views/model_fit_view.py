from configparser import ConfigParser
from pathlib import Path
from tkinter import NSEW, EW, E, StringVar
from typing import Optional

from AU_recognizer.core.user_interface import ScrollableFrame, CustomButton, CustomCheckBox
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.user_interface.widgets.complex_widget import ComboLabel, RadioList
from AU_recognizer.core.util import (nect_config, CONFIG, MODEL_FOLDER, R_DETAIL, R_COARSE,
                                     logger, i18n, MF_SELECT_IMG, MF_MODEL, MF_SAVE_IMAGES, MF_SAVE_CODES, MF_SAVE_MESH,
                                     MF_FIT_MODE)


class ModelFitView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.scrollFrame = ScrollableFrame(self)  # add a new scrollable frame.
        self.scrollFrame.rowconfigure(0, weight=1)
        self.scrollFrame.columnconfigure(0, weight=1)
        self._project_info: Optional[ConfigParser] = None
        models = [x for x in Path(nect_config[CONFIG][MODEL_FOLDER]).iterdir() if x.is_dir()]
        self.model_combobox = ComboLabel(master=self.scrollFrame, label_text="m_combo",
                                         selected="EMOCA_v2_lr_mse_20",
                                         values=[str(file_name.stem) for file_name in models], state="readonly")
        self.save_images_text = StringVar()
        self.save_images = CustomCheckBox(master=self.scrollFrame, text="c_simages", textvariable=self.save_images_text,
                                          check_state=True)
        self.save_codes_text = StringVar()
        self.save_codes = CustomCheckBox(master=self.scrollFrame, text="c_scodes", textvariable=self.save_codes_text)
        self.save_mesh_text = StringVar()
        self.save_mesh = CustomCheckBox(master=self.scrollFrame, text="c_smesh", textvariable=self.save_mesh_text)
        self.fit_mode = RadioList(master=self.scrollFrame, list_title="mode_radio", default=R_DETAIL,
                                  data=[R_DETAIL, R_COARSE])

        self.select_images_button_label = StringVar()
        self.select_images_button: Optional[CustomButton] = None
        self.update_language()

    def update_language(self):
        logger.debug("update language in ModelFit view")
        self.model_combobox.update_language()
        self.save_images.update_language()
        self.save_codes.update_language()
        self.save_mesh.update_language()
        self.fit_mode.update_language()
        self.select_images_button_label.set(i18n.project_actions_fit[MF_SELECT_IMG])

    def create_view(self):
        logger.debug("create view in ModelFit view")
        self.scrollFrame.grid(row=0, column=0, sticky=NSEW, padx=10, pady=5)
        # Combo box for model selection
        self.model_combobox.create_view()
        # Checkboxes for options
        self.fit_mode.create_view()
        # Button to select images
        self.select_images_button = CustomButton(self.scrollFrame, textvariable=self.select_images_button_label)

    def __update_view(self):
        logger.debug("update view in ModelFit view")
        self.model_combobox.grid(row=0, column=0, sticky=EW, pady=5)
        self.save_images.grid(row=1, column=0, sticky=EW, pady=5)
        self.save_codes.grid(row=2, column=0, sticky=EW, pady=5)
        self.save_mesh.grid(row=3, column=0, sticky=EW, pady=5)
        self.fit_mode.grid(row=4, column=0, sticky=EW, pady=5)
        self.select_images_button.grid(row=5, column=0, sticky=E, pady=5)

    def update_selected_project(self, data: Optional[ConfigParser] = None):
        logger.debug("update selected project in scan view")
        self._project_info = data
        self.__update_view()

    def get_form(self):
        return {
            MF_MODEL: self.model_combobox.get_value(),
            MF_SAVE_IMAGES: self.save_images.get(),
            MF_SAVE_CODES: self.save_codes.get(),
            MF_SAVE_MESH: self.save_mesh.get(),
            MF_FIT_MODE: self.fit_mode.get_value()
        }
