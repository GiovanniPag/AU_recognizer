from pathlib import Path
from tkinter import StringVar, EW, NSEW
from typing import Optional

from AU_recognizer.core.user_interface import CustomButton, CustomLabel
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import FV_OPEN_S, logger, i18n, FV_FILE, FV_SUFFIX, FV_NO, P_PATH, FV_TYPE, FV_U, FV_F


class SelectedFileView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.data = None
        self._file_label = StringVar()
        self._suffix_label = StringVar()
        self._file_label_info = StringVar()
        self._suffix_label_info = StringVar()
        self._no_file = StringVar()

        self._open_s_label = StringVar()
        self._open_s: Optional[CustomButton] = None

        self._buttons = {
            FV_OPEN_S: lambda: self._open_s
        }
        self.update_language()

    def update_language(self):
        logger.debug("update language in selected file view")
        self._file_label_info.set(i18n.selected_file_view[FV_FILE])
        self._suffix_label_info.set(i18n.selected_file_view[FV_SUFFIX])
        self._no_file.set(i18n.selected_file_view[FV_NO])
        self._open_s_label.set(i18n.selected_file_view[FV_OPEN_S])
        if self.data:
            path = Path(self.data[P_PATH])
            self._suffix_label.set(
                str(path.suffix) + (", " if str(path.suffix) else "") + (i18n.selected_file_view[FV_TYPE].get(
                    str(path.suffix), i18n.selected_file_view[FV_TYPE][FV_U])) if path.is_file() else
                i18n.selected_file_view[FV_TYPE][FV_F])

    def create_view(self):
        logger.debug("create view in selected file view")
        self._open_s = CustomButton(self, textvariable=self._open_s_label, command=...)

    def __update_view(self):
        logger.debug("update view in selected file view")
        for widget in self.winfo_children():
            if isinstance(widget, CustomLabel):
                widget.destroy()
            else:
                widget.grid_forget()
        if self.data:
            CustomLabel(self, textvariable=self._file_label_info).grid(column=0, row=0, sticky=EW)
            CustomLabel(self, textvariable=self._file_label).grid(column=1, row=0, sticky=EW)
            CustomLabel(self, textvariable=self._suffix_label_info).grid(column=0, row=1, sticky=EW)
            CustomLabel(self, textvariable=self._suffix_label).grid(column=1, row=1, sticky=EW)
            # update buttons
            self._open_s.grid(column=0, row=2, sticky=EW)
        else:
            CustomLabel(self, textvariable=self._no_file).grid(column=0, row=0, sticky=NSEW)

    def set_command(self, btn_name, command):
        logger.debug(f"ModelFit action view set command {command} for {btn_name}")
        if btn_name in self._buttons.keys():
            button: CustomButton = self._buttons.get(btn_name)()
            button.configure(command=command)

    def update_selected_file(self, data: Optional[dict] = None):
        logger.debug("update selected file in selected file view")
        self.data = data
        if data:
            path = Path(data[P_PATH])
            self._file_label.set(str(path.name))
            self._suffix_label.set(
                str(path.suffix) + (", " if str(path.suffix) else "") + (i18n.selected_file_view[FV_TYPE].get(
                    str(path.suffix), i18n.selected_file_view[FV_TYPE][FV_U])) if path.is_file() else
                i18n.selected_file_view[FV_TYPE][FV_F])
        self.__update_view()
