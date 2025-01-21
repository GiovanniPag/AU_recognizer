from configparser import ConfigParser
from tkinter import StringVar, W, EW, NSEW
from typing import Optional

from AU_recognizer.core.user_interface import CustomLabel
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import logger, i18n, PV_PATH, PV_NAME, PV_NO, P_DONE, PV_DONE, PV_NOT_DONE, P_PATH, P_NAME


class ProjectInfoView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._project_name = None
        self._name_label_info = StringVar()
        self._path_label_info = StringVar()
        self._name_label = StringVar()
        self._path_label = StringVar()
        self._no_project = StringVar()
        self.update_language()

    def update_language(self):
        logger.debug("update language in project info view")
        self._path_label_info.set(i18n.selected_project_view[PV_PATH])
        self._name_label_info.set(i18n.selected_project_view[PV_NAME])
        self._no_project.set(i18n.selected_project_view[PV_NO])

    def create_view(self):
        logger.debug("create view in project info view")
        for widget in self.winfo_children():
            widget.destroy()
        if self._project_info:
            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            CustomLabel(self, textvariable=self._name_label_info).grid(column=0, row=0, sticky=W)
            CustomLabel(self, textvariable=self._name_label).grid(column=1, row=0, sticky=EW)
            CustomLabel(self, textvariable=self._path_label_info).grid(column=0, row=1, sticky=W)
            CustomLabel(self, textvariable=self._path_label).grid(column=1, row=1, sticky=EW)
        else:
            CustomLabel(self, textvariable=self._no_project).grid(column=0, row=0, sticky=NSEW)

    def __step_done_set_label(self, label, info_to_check):
        logger.debug(f"set {label} of {info_to_check} as done or not")
        if self._project_info.getboolean(info_to_check, P_DONE, fallback=False):
            label.set(i18n.selected_project_view[PV_DONE])
        else:
            label.set(i18n.selected_project_view[PV_NOT_DONE])

    def update_selected_project(self, data: ConfigParser):
        logger.debug("update selected project in project info view")
        self._project_info = data
        if self._project_info:
            self._project_name = str(self._project_info.sections()[0])
            self._path_label.set(str(self._project_info[self._project_name][P_PATH]))
            self._name_label.set(self._project_info[self._project_name][P_NAME])
        self.create_view()
