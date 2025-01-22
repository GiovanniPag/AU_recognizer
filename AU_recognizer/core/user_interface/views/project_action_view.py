from AU_recognizer.core.user_interface import CustomTabview
from AU_recognizer.core.user_interface.views.au_recognition_view import AURecognitionView
from AU_recognizer.core.user_interface.views.model_fit_view import ModelFitView
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import logger, i18n, PA_NAME, PA_DISABLED, PA_NORMAL


class ProjectActionView(CustomTabview, View):
    def __init__(self, master=None):
        super().__init__(master)
        CustomTabview.__init__(self, master)
        self._aur_frame_name = i18n.project_actions_au_rec[PA_NAME]
        self._mf_frame_name = i18n.project_actions_fit[PA_NAME]
        self.master = master
        self._project_info = None
        self._model_fit_frame = None
        self._au_recognition_frame = None

    def update_language(self):
        logger.debug(f"update language in project action view")
        self.rename(old_name=self._mf_frame_name, new_name=i18n.project_actions_fit[PA_NAME])
        self.rename(old_name=self._aur_frame_name, new_name=i18n.project_actions_au_rec[PA_NAME])
        self._aur_frame_name = i18n.project_actions_au_rec[PA_NAME]
        self._mf_frame_name = i18n.project_actions_fit[PA_NAME]
        self._model_fit_frame.update_language()
        self._au_recognition_frame.update_language()

    def __tabs_change_state(self, new_state):
        logger.debug(f"set all action tabs in state: {new_state}")
        self.configure(state=new_state)

    def create_view(self):
        logger.debug(f"create view in project action view")
        _mf_frame = self.add(name=i18n.project_actions_fit[PA_NAME])
        self._model_fit_frame = ModelFitView(_mf_frame)
        self._model_fit_frame.grid(row=0, column=0)
        _aur_frame = self.add(name=i18n.project_actions_au_rec[PA_NAME])
        self._au_recognition_frame = AURecognitionView(_aur_frame)
        self._au_recognition_frame.grid(row=0, column=0)
        self.__update_view()

    def __update_view(self):
        logger.debug(f"update view in project action view")
        if not self._project_info:
            self.__tabs_change_state(PA_DISABLED)
        else:
            self.__tabs_change_state(PA_NORMAL)
            self.set(i18n.project_actions_fit[PA_NAME])

    def print_children(self, widget, depth=0):
        """Recursively print all children of a widget."""
        indent = "  " * depth  # Indentation for better visualization
        print(f"{indent}{widget}")
        for child in widget.winfo_children():
            self.print_children(child, depth + 1)

    def bind_controllers(self, model_fit_controller, au_recognition_controller):
        logger.debug(f"bind controllers in project action view")
        self.__bind_controller(controller=model_fit_controller, view=self._model_fit_frame)
        self.__bind_controller(controller=au_recognition_controller, view=self._au_recognition_frame)

    @staticmethod
    def __bind_controller(controller, view: View):
        logger.debug(f"project action bind {controller.__class__} to {view.__class__}")
        controller.bind(view)

    def update_selected_project(self, data=None):
        logger.debug(f"update selected project in project action view")
        self._project_info = data
        self.__update_view()
