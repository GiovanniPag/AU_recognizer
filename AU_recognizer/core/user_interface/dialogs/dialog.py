import configparser
import copy
from abc import abstractmethod
from pprint import pprint
from tkinter import StringVar, EW, E, NSEW, END, RIGHT, X, BOTH, BooleanVar, LEFT
from tkinter.ttk import Treeview

import numpy as np
import torch
from matplotlib import cm

from AU_recognizer.core.projects.emoca_fitter import emoca_fit
from AU_recognizer.core.user_interface import CustomToplevel, CustomLabel, CustomEntry, CustomButton, CustomFrame, \
    ScrollableFrame, CustomCheckBox, CustomTabview, ThemeManager
from AU_recognizer.core.user_interface.widgets.complex_widget import EntryButton, ComboLabel, \
    NumberPicker, ColorPickerLabel
from AU_recognizer.core.util import retrieve_files_from_path, OBJ
from AU_recognizer.core.util.config import logger, nect_config, write_config
from AU_recognizer.core.util.constants import *
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.util.utility_functions import check_name, check_file_name, lighten_color, gray_to_hex
from gdl.models.DecaFLAME import FLAME_mediapipe
from gdl_apps.EMOCA.utils.load import load_model


class Dialog(CustomToplevel):
    def show(self):
        logger.debug(f"show dialog")
        self.create_view()
        self.protocol("WM_DELETE_WINDOW", self.dismiss)  # intercept close button
        self.transient(self.master)  # dialog window is related to main
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


class SelectFitImageDialog(Dialog):
    def __init__(self, master, data, project, title=i18n.project_actions_fit[PA_NAME]):
        super().__init__(master)
        super().title(title)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])
        self.selected_images = []
        self.fit_data = data
        self.main_frame = CustomFrame(self)
        self.bottom_frame = CustomFrame(self.main_frame)
        self.image_treeview = Treeview(self.main_frame, columns=(TD_IMAGES, TD_FITTED), selectmode='extended')
        # Variable for checkbox state
        self.hide_fitted_var = BooleanVar(value=False)
        # Create the checkbox
        self.hide_fitted_checkbox = CustomCheckBox(
            self.main_frame,
            text=i18n.im_sel_dialog[TD_HIDE_FITTED],
            variable=self.hide_fitted_var,
            command=self.populate_image_treeview  # Refresh the Treeview when toggled
        )
        # Entry widget for filter text
        self.filter_var = StringVar()  # StringVar to hold the filter input
        self.filter_entry = CustomEntry(
            self.main_frame,
            textvariable=self.filter_var,
            validate="key"
        )
        self.filter_entry.bind("<KeyRelease>", self.filter_images)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=True)
        # Treeview for images
        self.image_treeview.column(TD_IMAGES, anchor=T_CENTER, minwidth=300, width=300)
        self.image_treeview.column(TD_FITTED, anchor=T_CENTER, minwidth=100, width=100, stretch=False)
        self.image_treeview.heading(TD_IMAGES, text=i18n.im_sel_dialog[T_COLUMNS][TD_IMAGES])
        self.image_treeview.heading(TD_FITTED, text=i18n.im_sel_dialog[T_COLUMNS][TD_FITTED])
        # select mode
        self.image_treeview["selectmode"] = "extended"
        # displayColumns
        self.image_treeview["displaycolumns"] = [TD_IMAGES, TD_FITTED]
        # show
        self.image_treeview["show"] = "headings"
        self.image_treeview.pack(fill=BOTH, expand=True)
        # Bottom frame for buttons
        self.bottom_frame.pack(fill=X, expand=True)
        self.filter_entry.pack(fill=X, padx=5, pady=5)  # Filter text box
        # Add the checkbox above the Treeview
        self.hide_fitted_checkbox.pack(anchor='w')
        # Treeview for images
        # Buttons
        close_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=RIGHT, padx=5, pady=5)
        fit_selected_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_SEL_BUTTON],
                                           command=self.fit_selected)
        fit_selected_button.pack(side=RIGHT, padx=5, pady=5)
        fit_all_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_ALL_BUTTON],
                                      command=self.fit_all)
        fit_all_button.pack(side=RIGHT, padx=5, pady=5)
        # Populate the treeview with images
        self.populate_image_treeview()

    def filter_images(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_image_treeview()  # Repopulate treeview with filter applied

    def populate_image_treeview(self):
        logger.debug(f"{self.__class__.__name__} populate image treeview")
        # Clear existing entries in the Treeview
        for item in self.image_treeview.get_children():
            self.image_treeview.delete(item)
        # List all image files in the folder_path
        images_path = Path(self.project[self._project_name][P_PATH]) / F_INPUT
        # List of desired image suffixes
        image_suffixes = ['.jpg', '.png', '.bmp']
        # Get all image files with the specified suffixes
        image_files = [file for file in images_path.iterdir() if file.suffix.lower() in image_suffixes]
        # Sort images by name (alphabetically)
        image_files.sort(key=lambda x: x.name.lower())
        # Get the filter value from the Entry widget
        filter_text = self.filter_var.get().lower()
        for idx, image in enumerate(image_files):
            tag = 'even' if idx % 2 == 0 else 'odd'
            output_folder = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / self.fit_data[MF_MODEL]
            fit_status = i18n.im_sel_dialog["data"]["fitted"] if output_folder.exists() and (
                any(d.is_dir() and d.name.startswith(image.stem) for d in output_folder.iterdir())) else \
                i18n.im_sel_dialog["data"]["not_fitted"]
            # Check the state of the checkbox to decide whether to add the image
            if (not self.hide_fitted_var.get() or fit_status == i18n.im_sel_dialog["data"]["not_fitted"]) and \
                    (filter_text in image.name.lower()):
                self.image_treeview.insert('', END, values=(image.name, fit_status, image), tags=(tag,))
        # Define tags and styles for alternating row colors
        self.image_treeview.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.image_treeview.tag_configure('odd',
                                          background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                              ThemeManager.theme["CustomFrame"]["fg_color"])), 10))

    def fit_all(self):
        logger.debug(f"{self.__class__.__name__} fit all images")
        emoca_fit(fit_data=self.fit_data, images_to_fit=Path(self.project[self._project_name][P_PATH]) / F_INPUT,
                  project_data=self.project)
        logger.debug("<<UpdateTreeSmall>> event generation")
        self.master.event_generate("<<UpdateTreeSmall>>")

    def fit_selected(self):
        logger.debug(f"{self.__class__.__name__} fit selected images")
        selected_items = self.image_treeview.selection()
        images_to_fit = []
        for item in selected_items:
            image_name = self.image_treeview.item(item, 'values')[0]
            images_to_fit.append(Path(self.project[self._project_name][P_PATH]) / F_INPUT / image_name)
        if images_to_fit:
            emoca_fit(fit_data=self.fit_data, images_to_fit=images_to_fit, project_data=self.project)
        else:
            logger.info("No images where selected.")
            data = i18n.project_message["no_images_selected"]
            DialogMessage(master=self,
                          title=data[I18N_TITLE],
                          message=data[I18N_MESSAGE],
                          detail=data[I18N_DETAIL],
                          icon=INFORMATION_ICON).show()
        logger.debug("<<UpdateTreeSmall>> event generation")
        self.master.event_generate("<<UpdateTreeSmall>>")

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value")
        pass

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        pass


class SelectMeshDialog(Dialog):
    def __init__(self, master, project, title=i18n.project_actions_au_rec[PA_NAME]):
        super().__init__(master)
        super().title(title)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])
        self.selected_images = []
        self.main_frame = CustomFrame(self)
        self.neutral_frame = CustomFrame(self.main_frame)
        self.compare_frame = CustomFrame(self.main_frame)
        self.bottom_frame = CustomFrame(self)
        self.mesh_treeview_neutral = Treeview(self.neutral_frame, columns=TD_MESH_N, selectmode='browse')
        # Entry widget for filter text
        self.filter_var_neutral = StringVar()  # StringVar to hold the filter input
        self.filter_entry_neutral = CustomEntry(
            self.neutral_frame,
            textvariable=self.filter_var_neutral,
            validate="key"
        )
        self.filter_entry_neutral.bind("<KeyRelease>", self.filter_mesh_neutral)

        self.mesh_treeview_compare = Treeview(self.compare_frame, columns=TD_MESH_C, selectmode='extended')
        # Entry widget for filter text
        self.filter_var_compare = StringVar()  # StringVar to hold the filter input
        self.filter_entry_compare = CustomEntry(
            self.compare_frame,
            textvariable=self.filter_var_compare,
            validate="key"
        )
        self.filter_entry_compare.bind("<KeyRelease>", self.filter_mesh_compare)
        # List all image files in the folder_path
        models_path = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT
        # List of required files
        required_files = ['mesh_coarse.obj', 'pose.npy', 'shape.npy', 'tex.npy', 'detail.npy', 'exp.npy']
        # List all directories inside mesh_path and check for the required files
        self.directories_useful = []
        for directory in models_path.rglob('*'):  # rglob searches recursively:
            if directory.is_dir() and self.contains_required_files(directory, required_files):
                self.directories_useful.append(directory)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=True)
        # neutral_left_frame
        self.neutral_frame.pack(side=LEFT, fill=BOTH, expand=True)
        # Treeview for images
        self.mesh_treeview_neutral.column(TD_MESH_N, anchor=T_CENTER, minwidth=300, width=300)
        self.mesh_treeview_neutral.heading(TD_MESH_N, text=i18n.mesh_sel_dialog[T_COLUMNS][TD_MESH_N])
        # select mode
        self.mesh_treeview_neutral["selectmode"] = "browse"
        # displayColumns
        self.mesh_treeview_neutral["displaycolumns"] = [TD_MESH_N]
        # show
        self.mesh_treeview_neutral["show"] = "headings"
        self.mesh_treeview_neutral.pack(fill=BOTH, expand=True)
        self.filter_entry_neutral.pack(fill=X, padx=5, pady=5)  # Filter text box
        # compare right frame
        self.compare_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        # Treeview for images
        self.mesh_treeview_compare.column(TD_MESH_C, anchor=T_CENTER, minwidth=300, width=300)
        self.mesh_treeview_compare.heading(TD_MESH_C, text=i18n.mesh_sel_dialog[T_COLUMNS][TD_MESH_C])
        # select mode
        self.mesh_treeview_compare["selectmode"] = "extended"
        # displayColumns
        self.mesh_treeview_compare["displaycolumns"] = [TD_MESH_C]
        # show
        self.mesh_treeview_compare["show"] = "headings"
        self.mesh_treeview_compare.pack(fill=BOTH, expand=True)
        self.filter_entry_compare.pack(fill=X, padx=5, pady=5)  # Filter text box
        # Bottom frame for buttons
        self.bottom_frame.pack(fill=X, expand=True)
        # Buttons
        close_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=RIGHT, padx=5, pady=5)
        compare_selected_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_COMPARE_SEL_BUTTON],
                                               command=self.compare_selected)
        compare_selected_button.pack(side=RIGHT, padx=5, pady=5)
        # Populate the treeview with images
        self.populate_neutral_treeview()
        self.populate_compare_treeview()

    def filter_mesh_neutral(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_neutral_treeview()  # Repopulate treeview with filter applied

    def filter_mesh_compare(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_compare_treeview()  # Repopulate treeview with filter applied

    # Function to check if all required files are in a directory
    @staticmethod
    def contains_required_files(directory, required_files):
        return all((directory / file).exists() for file in required_files)

    def populate_neutral_treeview(self):
        logger.debug(f"{self.__class__.__name__} populate mesh neutral treeview")
        # Clear existing entries in the Treeview
        for item in self.mesh_treeview_neutral.get_children():
            self.mesh_treeview_neutral.delete(item)
        # Get the filter value from the Entry widget
        filter_text = self.filter_var_neutral.get().lower()
        for idx, folder in enumerate(self.directories_useful):
            tag = 'even' if idx % 2 == 0 else 'odd'
            # Check the state of the checkbox to decide whether to add the image
            if filter_text == "" or filter_text in folder.name.lower():
                self.mesh_treeview_neutral.insert('', END, values=(folder.name, folder), tags=(tag,))
        # Define tags and styles for alternating row colors
        self.mesh_treeview_neutral.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.mesh_treeview_neutral.tag_configure('odd',
                                                 background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                     ThemeManager.theme["CustomFrame"]["fg_color"])), 10))

    def populate_compare_treeview(self):
        logger.debug(f"{self.__class__.__name__} populate mesh compare treeview")
        # Clear existing entries in the Treeview
        for item in self.mesh_treeview_compare.get_children():
            self.mesh_treeview_compare.delete(item)
        # Get the filter value from the Entry widget
        filter_text = self.filter_var_compare.get().lower()
        for idx, folder in enumerate(self.directories_useful):
            tag = 'even' if idx % 2 == 0 else 'odd'
            # Check the state of the checkbox to decide whether to add the image
            if filter_text == "" or filter_text in folder.name.lower():
                self.mesh_treeview_compare.insert('', END, values=(folder.name, folder), tags=(tag,))
        # Define tags and styles for alternating row colors
        self.mesh_treeview_compare.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.mesh_treeview_compare.tag_configure('odd',
                                                 background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                     ThemeManager.theme["CustomFrame"]["fg_color"])), 10))

    def compare_selected(self):
        logger.debug(f"{self.__class__.__name__} fit selected images")
        selected_items = self.mesh_treeview_neutral.selection()
        neutral_face = None
        for item in selected_items:
            neutral_face = self.mesh_treeview_neutral.item(item, 'values')[1]
        if not neutral_face:
            logger.info("No neutral face where selected.")
            data = i18n.project_message["no_neutral_selected"]
            DialogMessage(master=self,
                          title=data[I18N_TITLE],
                          message=data[I18N_MESSAGE],
                          detail=data[I18N_DETAIL],
                          icon=INFORMATION_ICON).show()
            return
        selected_items = self.mesh_treeview_compare.selection()
        faces_compare = []
        for item in selected_items:
            compare_face = self.mesh_treeview_compare.item(item, 'values')[1]
            faces_compare.append(compare_face)
        if not faces_compare:
            logger.info("No compare face where selected.")
            data = i18n.project_message["no_compare_selected"]
            DialogMessage(master=self,
                          title=data[I18N_TITLE],
                          message=data[I18N_MESSAGE],
                          detail=data[I18N_DETAIL],
                          icon=INFORMATION_ICON).show()
            return
        self.compare_faces(neutral_face_path=neutral_face, compare_list_paths=faces_compare)
        logger.debug("<<UpdateTreeSmall>> event generation")
        self.master.event_generate("<<UpdateTreeSmall>>")

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value")
        pass

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        pass

    def compare_faces(self, neutral_face_path, compare_list_paths):
        logger.debug(f"compare {neutral_face_path} with list {compare_list_paths}")
        output_path = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / F_COMPARE
        output_path.mkdir(parents=True, exist_ok=True)
        neutral_face_path = Path(neutral_face_path)
        neutral_obj = OBJ(filepath=neutral_face_path / "mesh_coarse.obj")
        n_verts = neutral_obj.get_vertices()
        neutral_pose = np.load(neutral_face_path / 'pose.npy')
        # load emoca model
        path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER])
        model_name = "EMOCA_v2_lr_mse_20"
        mode = "detail"
        # 0) clear memory
        torch.cuda.empty_cache()  # Clear any cached GPU memory
        # 1) Load the model
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()
        for face_path in compare_list_paths:
            face_path = Path(face_path)
            f_exp = np.load(face_path / 'exp.npy')
            # f_pose = np.load(face_path / 'pose.npy')
            f_shape = np.load(face_path / 'shape.npy')
            shape = torch.from_numpy(f_shape)
            exp = torch.from_numpy(f_exp)
            pose = torch.from_numpy(neutral_pose)
            shape = shape.unsqueeze(0)  # Shape: [1, 100]
            exp = exp.unsqueeze(0)  # Shape: [1, 50]
            pose = pose.unsqueeze(0)  # Shape: [1, 6]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shape = shape.to(device)
            exp = exp.to(device)
            pose = pose.to(device)
            if not isinstance(emoca.deca.flame, FLAME_mediapipe):
                verts, landmarks2d, landmarks3d = emoca.deca.flame(shape_params=shape, expression_params=exp,
                                                                   pose_params=pose)
            else:
                verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = emoca.deca.flame(shape_params=shape,
                                                                                          expression_params=exp,
                                                                                          pose_params=pose)
            face_verts = verts.squeeze(0)  # Remove the batch dimension
            face_verts_np = face_verts.cpu().numpy()  # Convert PyTorch tensor to NumPy array
            diffs = np.linalg.norm(face_verts_np - n_verts, axis=1)  # Shape: [5023]
            normalized_diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))
            colormap = cm.get_cmap("plasma")  # Choose a colormap
            vertex_colors = colormap(normalized_diffs)[:, :3]  # RGB values for each vertex
            compare_obj = copy.deepcopy(neutral_obj)
            compare_obj.set_vertex_colors(vertex_colors)
            compare_obj.save(output_path / f"{face_path.name}_with_heatmap.obj")
        self.log_face_comparisons(neutral_face_path, compare_list_paths)

    def log_face_comparisons(self, neutral_face_path, compare_list_paths):
        logger.debug(f"Logging comparison between {neutral_face_path} and {compare_list_paths}")
        output_path = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / F_COMPARE
        output_path.mkdir(parents=True, exist_ok=True)
        # Set up the log file path
        log_file = output_path / "face_comparisons.ini"
        # Create configparser object to write to ini file
        config = configparser.ConfigParser()
        # Load the existing file if it exists, to avoid overwriting previous logs
        if log_file.exists():
            config.read(log_file)
        # Create a new section for the neutral face comparison
        neutral_face_name = neutral_face_path.name
        config[neutral_face_name] = {}
        # Log the neutral face's comparison to each face in compare_list_paths
        for i, compare_face_path in enumerate(compare_list_paths):
            compare_face_name = Path(compare_face_path).name
            config[neutral_face_name][f"compared_face_{i + 1}"] = str(compare_face_name)
        # Save the log to the .ini file
        with open(log_file, 'w') as log:
            config.write(log)
        logger.debug(f"Comparison log saved to {log_file}")

class SettingsDialog(Dialog):
    def __init__(self, master, page=""):
        super().__init__(master)
        self.master = master
        # Create the notebook
        self.notebook = CustomTabview(self)

        # Create frames for the tabs
        self.general_frame = self.notebook.add(name=i18n.settings_dialog[GENERAL_TAB])
        self.viewer_frame = self.notebook.add(name=i18n.settings_dialog[VIEWER_TAB])
        self.startpage = page
        # Create the main frame
        self._i18n_path = EntryButton(master=self.general_frame, label_text="i18n_path",
                                      entry_text=nect_config[CONFIG][I18N_PATH])
        self._logger_path = EntryButton(master=self.general_frame, label_text="logger_path",
                                        entry_text=nect_config[CONFIG][LOGGER_PATH])
        self._log_folder = EntryButton(master=self.general_frame, label_text="log_folder",
                                       entry_text=nect_config[CONFIG][LOG_FOLDER])
        self._projects_folder = EntryButton(master=self.general_frame, label_text="projects_folder",
                                            entry_text=nect_config[CONFIG][PROJECTS_FOLDER])
        self._path_to_model = EntryButton(master=self.general_frame, label_text="path_to_model",
                                          entry_text=nect_config[CONFIG][MODEL_FOLDER])
        self.language = nect_config[CONFIG][LANGUAGE]
        self.i18n_json_files = retrieve_files_from_path(path=Path(nect_config[CONFIG][I18N_PATH]), file_type="*.json")
        self.languages_combobox = ComboLabel(master=self.general_frame, label_text="l_combo",
                                             values=[str(file_name.stem) for file_name in self.i18n_json_files],
                                             selected=self.language)
        # viewer tab
        self._fill_color = ColorPickerLabel(master=self.viewer_frame, label_text="fill_color",
                                            default=nect_config[VIEWER][FILL_COLOR],
                                            on_change=lambda: self.save_viewer_config())
        self._line_color = ColorPickerLabel(master=self.viewer_frame, label_text="line_color",
                                            default=nect_config[VIEWER][LINE_COLOR],
                                            on_change=lambda: self.save_viewer_config())
        self._canvas_color = ColorPickerLabel(master=self.viewer_frame, label_text="canvas_color",
                                              default=nect_config[VIEWER][CANVAS_COLOR],
                                              on_change=lambda: self.save_viewer_config())
        self._point_color = ColorPickerLabel(master=self.viewer_frame, label_text="point_color",
                                             default=nect_config[VIEWER][POINT_COLOR],
                                             on_change=lambda: self.save_viewer_config())
        self._point_size = NumberPicker(master=self.viewer_frame, label_text="point_size",
                                        default=nect_config[VIEWER][POINT_SIZE], min_value=1, max_value=20, increment=1,
                                        is_float=False,
                                        on_change=lambda: self.save_viewer_config())
        self._ground_color = ColorPickerLabel(master=self.viewer_frame, label_text="ground_color",
                                              default=nect_config[VIEWER][GROUND_COLOR],
                                              on_change=lambda: self.save_viewer_config())
        self._sky_color = ColorPickerLabel(master=self.viewer_frame, label_text="sky_color",
                                           default=nect_config[VIEWER][SKY_COLOR],
                                           on_change=lambda: self.save_viewer_config())
        self._moving_step = NumberPicker(master=self.viewer_frame, label_text="moving_step",
                                         default=nect_config[VIEWER][MOVING_STEP], min_value=0.01, max_value=1,
                                         increment=0.01, is_float=True,
                                         on_change=lambda: self.save_viewer_config())

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view method")
        self.grid_columnconfigure(0, weight=1)
        self.notebook.grid(row=0, column=0, sticky=NSEW, columnspan=2, padx=10)
        # Add frames to notebook as tabs
        self.__create_general_tab()
        self.__create_viewer_tab()
        save_button = CustomButton(self, text=i18n.dialog_buttons[I18N_SAVE_BUTTON], command=self.save_config)
        save_button.grid(row=1, column=0, sticky=E, padx=10, pady=10)
        close_button = CustomButton(self, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.grid(row=1, column=1, sticky=EW, padx=10, pady=10)
        if self.startpage == "viewer":
            self.notebook.set(i18n.settings_dialog[VIEWER_TAB])

    def save_config(self):
        logger.debug(f"{self.__class__.__name__} save config")
        # Store current values
        new_general_values = {
            LANGUAGE: self.languages_combobox.get_value(),
            I18N_PATH: self._i18n_path.get_value(),
            LOGGER_PATH: self._logger_path.get_value(),
            LOG_FOLDER: self._log_folder.get_value(),
            PROJECTS_FOLDER: self._projects_folder.get_value(),
            MODEL_FOLDER: self._path_to_model.get_value()
        }
        changed = i18n.change_language(new_general_values[LANGUAGE])
        if changed:
            logger.debug("SettingsDialog <<LanguageChange>> event generation")
            self.master.event_generate("<<LanguageChange>>")
        for key, value in new_general_values.items():
            if key != LANGUAGE and Path(value).exists():
                nect_config[CONFIG][key] = value
        new_viewer_value = {
            FILL_COLOR: self._fill_color.get_value(),
            LINE_COLOR: self._line_color.get_value(),
            CANVAS_COLOR: self._canvas_color.get_value(),
            POINT_COLOR: self._point_color.get_value(),
            GROUND_COLOR: self._ground_color.get_value(),
            SKY_COLOR: self._sky_color.get_value(),
            POINT_SIZE: self._point_size.get_value(),
            MOVING_STEP: self._moving_step.get_value()
        }
        for key, value in new_viewer_value.items():
            if str(nect_config[VIEWER][key]) != str(value):
                nect_config[VIEWER][key] = str(value)

        write_config()
        self.close()

    def save_viewer_config(self):
        new_viewer_value = {
            FILL_COLOR: self._fill_color.get_value(),
            LINE_COLOR: self._line_color.get_value(),
            CANVAS_COLOR: self._canvas_color.get_value(),
            POINT_COLOR: self._point_color.get_value(),
            GROUND_COLOR: self._ground_color.get_value(),
            SKY_COLOR: self._sky_color.get_value(),
            POINT_SIZE: self._point_size.get_value(),
            MOVING_STEP: self._moving_step.get_value()
        }
        for key, value in new_viewer_value.items():
            if str(nect_config[VIEWER][key]) != str(value):
                nect_config[VIEWER][key] = str(value)
        write_config()
        self.update_viewer()

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value")
        self.update_viewer()

    def update_viewer(self):
        self.master.event_generate("<<ViewerChange>>")

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        pass

    def __create_general_tab(self):
        self.general_frame.grid_columnconfigure(0, weight=1)
        self._i18n_path.grid(row=0, column=0, sticky=EW, columnspan=2, padx=10)
        self._i18n_path.create_view()
        self.languages_combobox.grid(row=1, column=0, sticky=EW, columnspan=2, padx=10)
        self.languages_combobox.create_view()
        self._logger_path.grid(row=2, column=0, sticky=EW, columnspan=2, padx=10)
        self._logger_path.create_view()
        self._log_folder.grid(row=3, column=0, sticky=EW, columnspan=2, padx=10)
        self._log_folder.create_view()
        self._projects_folder.grid(row=4, column=0, sticky=EW, columnspan=2, padx=10)
        self._projects_folder.create_view()
        self._path_to_model.grid(row=5, column=0, sticky=EW, columnspan=2, padx=10)
        self._path_to_model.create_view()

    def __create_viewer_tab(self):
        self.viewer_frame.grid_columnconfigure(0, weight=1)
        self._fill_color.grid(row=0, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._fill_color.create_view()
        self._line_color.grid(row=1, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._line_color.create_view()
        self._canvas_color.grid(row=2, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._canvas_color.create_view()
        self._ground_color.grid(row=4, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._ground_color.create_view()
        self._sky_color.grid(row=5, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._sky_color.create_view()
        self._point_color.grid(row=3, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._point_color.create_view()
        self._point_size.grid(row=6, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._point_size.create_view()
        self._moving_step.grid(row=7, column=0, sticky=EW, columnspan=2, padx=10, pady=5)
        self._moving_step.create_view()


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
