from abc import abstractmethod
from tkinter import StringVar

from AU_recognizer.core.user_interface import CustomToplevel, CustomLabel, CustomEntry, CustomButton
from AU_recognizer.core.util.config import logger
from AU_recognizer.core.util.constants import *
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.util.utility_functions import check_name, check_file_name


class Dialog(CustomToplevel):
    def show(self):
        logger.debug(f"show dialog")
        self.create_view()
        self.protocol("WM_DELETE_WINDOW", self.dismiss)  # intercept close button
        self.transient(self.master)  # type: ignore # dialog window is related to main
        self.wait_visibility()  # can't grab until window appears, so we wait
        self.grab_set()  # ensure all input goes to our window
        self.master.wait_window(self)  # block until window is destroyed
        return self.ask_value()

    @abstractmethod
    def ask_value(self):
        raise NotImplementedError

    @abstractmethod
    def dismiss_method(self):
        raise NotImplementedError

    @abstractmethod
    def create_view(self):
        raise NotImplementedError

    def close(self):
        logger.debug("Close Dialog")
        self.grab_release()
        self.destroy()

    def dismiss(self):
        logger.debug("Dismiss Dialog")
        self.dismiss_method()
        self.close()


class BaseDialog(Dialog):
    def __init__(self, master, i18n_data, validate_command, has_back=False):
        super().__init__(master)
        self.master = master
        self.i18n_data = i18n_data
        self.t = self.i18n_data[I18N_TITLE]
        self.name_var = StringVar()
        self.name_var.set("")
        self.has_back = has_back
        self.back = False
        self.name_label = self.i18n_data[I18N_NAME]
        self.name_tip = self.i18n_data[I18N_NAME_TIP]
        self.validate_command = validate_command

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value: {self.name_var.get()}")
        return i18n.dialog_buttons[I18N_BACK_BUTTON] if self.back else self.name_var.get()

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        self.name_var.set("")

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view method")
        self.title(self.t)
        self.resizable(False, False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        CustomLabel(self, image=FULL_QUESTION_ICON).grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7),
                                                         sticky="e")
        CustomLabel(self, text=self.name_label).grid(row=0, column=1, columnspan=1, pady=(7, 7), padx=(7, 7),
                                                     sticky="w")
        entry = CustomEntry(self, textvariable=self.name_var, validate='key',
                            validatecommand=(self.master.register(self.validate_command), '%P'))
        entry.bind("<Return>", lambda event: self.close())
        entry.grid(row=0, column=2, columnspan=3, pady=(7, 7), padx=(7, 7), sticky="we")
        CustomLabel(self, text=self.name_tip, wraplength=400).grid(row=1, column=1, columnspan=4,
                                                                   padx=(7, 7), sticky="we")
        CustomButton(self, text=i18n.dialog_buttons[I18N_CONFIRM_BUTTON], command=self.close).grid(row=2, column=2,
                                                                                                   pady=(7, 7),
                                                                                                   padx=(7, 7),
                                                                                                   sticky="e")
        c_c = 3
        if self.has_back:
            c_c = 4
            CustomButton(self, text=i18n.dialog_buttons[I18N_BACK_BUTTON], command=self.go_back).grid(row=2, column=3,
                                                                                                      pady=(7, 7),
                                                                                                      padx=(7, 7),
                                                                                                      sticky="e")
        CustomButton(self, text=i18n.dialog_buttons[I18N_CANCEL_BUTTON], command=self.dismiss).grid(row=2, column=c_c,
                                                                                                    pady=(7, 7),
                                                                                                    padx=(7, 7),
                                                                                                    sticky="e")
        entry.focus_set()

    def go_back(self):
        if self.has_back:
            logger.debug("Project Options Dialog Back method")
            self.back = True
            self.close()


class DialogProjectOptions(BaseDialog):
    def __init__(self, master, has_back=False):
        super().__init__(master=master, i18n_data=i18n.p_options_dialog, validate_command=check_name, has_back=has_back)


class DialogPathRename(BaseDialog):
    def __init__(self, master, has_back=False):
        super().__init__(master=master, i18n_data=i18n.p_rename_dialog, validate_command=check_file_name,
                         has_back=has_back)


class DialogMessage(Dialog):
    def __init__(self, master, title, message, detail, icon=INFORMATION_ICON):
        super().__init__(master)
        self.master = master
        self.icon = icon
        self.t = title
        self.message = message
        self.detail = detail

    def ask_value(self):
        pass

    def dismiss_method(self):
        pass

    def create_view(self):
        logger.debug("Message Dialog create view method")
        self.title(self.t)
        self.resizable(False, False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        CustomLabel(self, image=BASENAME_ICON + self.icon) \
            .grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7), sticky="e")
        CustomLabel(self, text=self.message) \
            .grid(row=0, column=1, columnspan=2, pady=(7, 7), padx=(7, 7), sticky="w")
        CustomLabel(self, text=self.detail).grid(row=1, column=1, columnspan=2, pady=(7, 7), padx=(7, 7), sticky="w")
        CustomButton(self, text=i18n.dialog_buttons[I18N_BACK_BUTTON], command=self.dismiss) \
            .grid(row=2, column=2, pady=(7, 7), padx=(7, 7), sticky="e")


class DialogAsk(Dialog):
    def __init__(self, master, title, message, detail,
                 options=None, dismiss_response=I18N_NO_BUTTON, icon=INFORMATION_ICON):
        super().__init__(master)
        self.master = master
        self.icon = icon
        self.t = title
        self.message = message
        self.detail = detail
        self.dismiss_response = dismiss_response
        if options is None:
            self.options = [I18N_NO_BUTTON, I18N_YES_BUTTON]
        else:
            self.options = options
        self.response_var = StringVar(value=dismiss_response)

    def ask_value(self):
        logger.debug("ask Dialog: " + self.response_var.get())
        return self.response_var.get()

    def dismiss_method(self):
        logger.debug("ask Dialog dismiss method")
        self.response_var.set(self.dismiss_response)

    def response(self, option):
        logger.debug("ask Dialog respond with: " + option)
        self.response_var.set(option)
        self.close()

    def create_view(self):
        logger.debug("ask Dialog create view method")
        self.title(self.t)
        self.resizable(False, False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        CustomLabel(self, image=BASENAME_ICON + self.icon) \
            .grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7), sticky="e")
        CustomLabel(self, text=self.message) \
            .grid(row=0, column=1, columnspan=(1 + len(self.options)), pady=(7, 7), padx=(7, 7), sticky="w")
        CustomLabel(self, text=self.detail).grid(row=1, column=1, columnspan=(1 + len(self.options)), pady=(7, 7),
                                                 padx=(7, 7), sticky="w")
        for index, option in enumerate(self.options):
            b = CustomButton(self, text=i18n.dialog_buttons[option], command=lambda opt=option: self.response(opt))
            b.grid(row=2, column=2 + index, pady=(7, 7), padx=(7, 7), sticky="e")
