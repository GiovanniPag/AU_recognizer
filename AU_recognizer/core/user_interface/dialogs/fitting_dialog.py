import logging
import sys
import threading
from tkinter import BOTH, BOTTOM, X, RIGHT, TOP

from AU_recognizer.core.model.model_manager import load_model_class
from AU_recognizer.core.user_interface.dialogs.dialog_util import open_message_dialog
from AU_recognizer.core.util.TextRedirector import TextLogger, TextRedirector

from AU_recognizer.core.user_interface import CustomFrame, CustomButton, CustomTextbox, CustomProgressBar
from AU_recognizer.core.user_interface.dialogs.dialog import Dialog
from AU_recognizer.core.util import (i18n, PA_NAME, logger, I18N_CLOSE_BUTTON, ERROR_ICON, MODEL)


class FitDialog(Dialog):
    def __init__(self, master, fit_data, images_to_fit, project_data, title=i18n.project_actions_fit[PA_NAME]):
        super().__init__(master)
        self.fitting_thread = None
        self.logger = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        super().title(title)
        self.master = master
        self.project_data = project_data
        self.fit_data = fit_data
        self.images_to_fit = images_to_fit
        self.main_frame = CustomFrame(self)
        self.progress = CustomProgressBar(self.main_frame)
        self.logbox = CustomTextbox(self.main_frame)
        self.bottom_frame = CustomFrame(self)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(side=TOP, fill=BOTH, expand=True)
        self.progress.pack(side=TOP, pady=10, padx=10, fill=X, expand=True)
        self.progress.set(0)
        self.logbox.pack(side=BOTTOM, fill=BOTH, expand=True, pady=10)
        self.bottom_frame.pack(side=BOTTOM, fill=X)
        close_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[
            I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=RIGHT, padx=5, pady=5)
        # Set up text tags for different types of messages
        self.logbox.tag_config("stdout", foreground="white")
        self.logbox.tag_config("stderr", foreground="red")
        self.logbox.tag_config("log", foreground="blue")
        # Redirect output streams
        sys.stdout = TextRedirector(self.logbox, "stdout")
        sys.stderr = TextRedirector(self.logbox, "stderr")
        # Set up logging
        self.logger = logging.getLogger("TkinterLogger")
        self.logger.setLevel(logging.DEBUG)
        self.handler = TextLogger(self.logbox)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.run_fitting_thread()

    def run_fitting(self):
        model_class = load_model_class(self.fit_data[MODEL])  # helper that imports dynamically
        result = model_class.fit(fit_data=self.fit_data, images_to_fit=self.images_to_fit,
                                 project_data=self.project_data, progress_callback=self.progress.set)
        self.on_fitting_complete(success=result)

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value")
        pass

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        pass

    def on_fitting_complete(self, success):
        def update_ui():
            if success:
                logger.info("Good job! The fitting task completed successfully.")
                open_message_dialog(master=self, message="fitting_success")
            else:
                logger.error("The fitting task failed.")
                open_message_dialog(master=self, message="fitting_failure", icon=ERROR_ICON)
            self.after(0, self.close)  # Ensures closing happens *after* the dialog is dismissed

        self.after(0, update_ui)  # Runs in main thread

    def run_fitting_thread(self):
        self.fitting_thread = threading.Thread(target=self.run_fitting, daemon=True)
        self.fitting_thread.start()

    def close(self):
        # Restore stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Remove logging handler
        if self.logger and self.handler:
            self.logger.removeHandler(self.handler)
            self.handler.close()
        # Optionally wait for the thread or just let it die since it's daemon
        # self.fitting_thread.join(timeout=1)
        super().close()