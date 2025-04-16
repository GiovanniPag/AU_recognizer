import logging
import sys
import threading
from pathlib import Path
from tkinter import BOTH, BOTTOM, X, RIGHT, TOP

import torch
from tqdm import auto

from AU_recognizer.core.projects.datasets.ImageTestDataset import TestData
from AU_recognizer.core.projects.utils.model import save_obj, save_images, save_codes, test
from AU_recognizer.core.projects.utils.utility import load_model
from AU_recognizer.core.user_interface.dialogs.dialog_util import open_message_dialog
from AU_recognizer.core.util.TextRedirector import TextLogger, TextRedirector

from AU_recognizer.core.user_interface import CustomFrame, CustomButton, CustomTextbox, CustomProgressBar
from AU_recognizer.core.user_interface.dialogs.dialog import Dialog
from AU_recognizer.core.util import (i18n, PA_NAME, logger, I18N_CLOSE_BUTTON, P_PATH, MODEL_FOLDER, CONFIG,
                                     nect_config, MF_MODEL, F_OUTPUT, MF_FIT_MODE, MF_SAVE_MESH, MF_SAVE_IMAGES,
                                     MF_SAVE_CODES, ERROR_ICON)


class FitDialog(Dialog):
    def __init__(self, master, fit_data, images_to_fit, project_data, title=i18n.project_actions_fit[PA_NAME]):
        super().__init__(master)
        self.fitting_thread = None
        self.logger = None
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
        handler = TextLogger(self.logbox)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.run_fitting_thread()

    def run_fitting(self):
        try:
            logger.info(f"emoca fit of {self.images_to_fit}, with {self.fit_data}, and {self.project_data}")

            if isinstance(self.images_to_fit, list):
                self.images_to_fit = [str(image.resolve()) for image in self.images_to_fit]
            elif isinstance(self.images_to_fit, Path):
                self.images_to_fit = str(self.images_to_fit.resolve())
            else:
                logger.error(f"{self.images_to_fit} are not a list or a path")
                self.on_fitting_complete(success=False)
                return
            _project_name = str(self.project_data.sections()[0])
            project_path = Path(self.project_data[_project_name][P_PATH])
            path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER])
            model_name = self.fit_data[MF_MODEL]
            output_folder = project_path / F_OUTPUT / model_name
            mode = self.fit_data[MF_FIT_MODE]

            torch.cuda.empty_cache()
            emoca, conf = load_model(path_to_models, model_name, mode)
            emoca.cuda()
            emoca.eval()

            dataset = TestData(self.images_to_fit, max_detection=20)

            for i in auto.tqdm(range(len(dataset))):
                try:
                    batch = dataset[i]
                except Exception as e:
                    logger.warning(f"error during fitting {dataset.imagepath_list[i]}: {e}")
                    continue

                vals, visdict = test(emoca, batch)
                current_bs = batch["image"].shape[0]

                for j in range(current_bs):
                    name = batch["image_name"][j]
                    sample_output_folder = Path(output_folder) / name
                    sample_output_folder.mkdir(parents=True, exist_ok=True)

                    if self.fit_data[MF_SAVE_MESH]:
                        save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
                    if self.fit_data[MF_SAVE_IMAGES]:
                        save_images(output_folder, name, visdict, with_detection=True, i=j)
                    if self.fit_data[MF_SAVE_CODES]:
                        save_codes(Path(output_folder), name, vals, i=j)
                self.progress.set((i + 1) / len(dataset))
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Fitting failed: {e}")
            self.on_fitting_complete(success=False)  # Failure
            return
        self.on_fitting_complete(success=True)

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
