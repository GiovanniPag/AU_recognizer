from configparser import ConfigParser
from tkinter import StringVar, W, EW, NSEW
from typing import Optional

from AU_recognizer.core.model.model_manager import list_models
from AU_recognizer.core.user_interface import CustomTabview, CustomFrame, CustomLabel
from AU_recognizer.core.user_interface.views.au_recognition_view import AURecognitionView
from AU_recognizer.core.user_interface.views.au_tag_view import AUTagView
from AU_recognizer.core.user_interface.views.model_fit_view import ModelFitView
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.user_interface.widgets.complex_widget import ComboLabel
from AU_recognizer.core.util import logger, i18n, PA_NAME, PA_DISABLED, PA_NORMAL, PV_NAME, PV_NO, P_NAME, nect_config, \
    MODEL, CONFIG


class ProjectActionView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.top_frame = CustomFrame(master=self)
        self._name_label_info = StringVar(value=i18n.selected_project_view[PV_NAME])
        self._no_project = StringVar(value=i18n.selected_project_view[PV_NO])
        self._name_label = StringVar()
        models = list_models()
        files = [str(file.name) for model in models for file in model.get_models_list()]
        self.model_combobox = ComboLabel(master=self.top_frame, label_text="m_combo",
                                         selected=nect_config[CONFIG][MODEL],
                                         values=files, state="readonly")
        self.tab_view = CustomTabview(master=self)
        self._aur_frame_name = i18n.project_actions_au_rec[PA_NAME]
        self._mf_frame_name = i18n.project_actions_fit[PA_NAME]
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._model_fit_frame = None
        self._au_recognition_frame = None
        self._au_tag_frame = None

    def update_language(self):
        logger.debug(f"update language in project action view")
        self.tab_view.rename(old_name=self._mf_frame_name, new_name=i18n.project_actions_fit[PA_NAME])
        self.tab_view.rename(old_name=self._aur_frame_name, new_name=i18n.project_actions_au_rec[PA_NAME])
        self.model_combobox.update_language()
        self._aur_frame_name = i18n.project_actions_au_rec[PA_NAME]
        self._mf_frame_name = i18n.project_actions_fit[PA_NAME]
        self._model_fit_frame.update_language()
        self._au_recognition_frame.update_language()
        self._au_tag_frame.update_language()
        self._name_label_info.set(i18n.selected_project_view[PV_NAME])
        self._no_project.set(i18n.selected_project_view[PV_NO])

    def __tabs_change_state(self, new_state):
        logger.debug(f"set all action tabs in state: {new_state}")
        self.tab_view.configure(state=new_state)

    def create_view(self):
        logger.debug(f"create view in project action view")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.top_frame.grid(column=0, row=0, sticky=EW)
        self.model_combobox.create_view()
        self.model_combobox.bind_combobox_event(self.model_change)
        self.tab_view.grid(column=0, row=1, sticky=NSEW)
        _mf_frame = self.tab_view.add(name=i18n.project_actions_fit[PA_NAME])
        _mf_frame.rowconfigure(0, weight=1)
        _mf_frame.columnconfigure(0, weight=1)
        self._model_fit_frame = ModelFitView(_mf_frame, model=nect_config[CONFIG][MODEL])
        self._model_fit_frame.grid(row=0, column=0, sticky="nsew")
        _aur_frame = self.tab_view.add(name=i18n.project_actions_au_rec[PA_NAME])
        _aur_frame.rowconfigure(0, weight=1)
        _aur_frame.columnconfigure(0, weight=1)
        self._au_recognition_frame = AURecognitionView(_aur_frame)
        self._au_recognition_frame.grid(row=0, column=0, sticky="nsew")
        _aut_frame = self.tab_view.add(name=i18n.project_actions_au_tag[PA_NAME])
        _aut_frame.rowconfigure(0, weight=1)
        _aut_frame.columnconfigure(0, weight=1)
        self._au_tag_frame = AUTagView(_aut_frame)
        self._au_tag_frame.grid(row=0, column=0, sticky="nsew")
        self.__update_view()

    def __update_view(self):
        logger.debug(f"update view in project action view")
        for widget in self.top_frame.winfo_children():
            if isinstance(widget, CustomLabel):
                widget.destroy()
        if not self._project_info:
            CustomLabel(self.top_frame, textvariable=self._no_project).grid(column=0, row=0, sticky=W)
            self.__tabs_change_state(PA_DISABLED)
        else:
            CustomLabel(self.top_frame, textvariable=self._name_label_info).grid(column=0, row=0, sticky=W, padx=10)
            CustomLabel(self.top_frame, textvariable=self._name_label).grid(column=1, row=0, sticky=W)
            self.model_combobox.grid(row=1, column=0, sticky=EW, columnspan=10, pady=5)
            self.__tabs_change_state(PA_NORMAL)
            self.tab_view.set(i18n.project_actions_fit[PA_NAME])

    def model_change(self, new_model=nect_config[CONFIG][MODEL]):
        logger.debug("model changed, update views")
        self._model_fit_frame.update_model(new_model)
        self._au_recognition_frame.update_model(new_model)
        self._au_tag_frame.update_model(new_model)

    def print_children(self, widget, depth=0):
        """Recursively print all children of a widget."""
        indent = "  " * depth  # Indentation for better visualization
        print(f"{indent}{widget}")
        for child in widget.winfo_children():
            self.print_children(child, depth + 1)

    def bind_controllers(self, model_fit_controller, au_recognition_controller, au_tag_controller):
        logger.debug(f"bind controllers in project action view")
        self.__bind_controller(controller=model_fit_controller, view=self._model_fit_frame)
        self.__bind_controller(controller=au_recognition_controller, view=self._au_recognition_frame)
        self.__bind_controller(controller=au_tag_controller, view=self._au_tag_frame)

    @staticmethod
    def __bind_controller(controller, view: View):
        logger.debug(f"project action bind {controller.__class__} to {view.__class__}")
        controller.bind(view)

    def update_selected_project(self, data=None):
        logger.debug(f"update selected project in project action view")
        self._project_info = data
        if self._project_info:
            self._name_label.set(self._project_info[str(self._project_info.sections()[0])][P_NAME])
        self.__update_view()
