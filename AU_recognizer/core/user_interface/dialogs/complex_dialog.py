import configparser
import copy
import re
from pathlib import Path
from tkinter import BooleanVar, StringVar, BOTH, BOTTOM, X, LEFT, Y, RIGHT, END, NSEW, E, EW, TOP
from tkinter.ttk import Treeview

import numpy as np
import torch
from matplotlib import cm

from AU_recognizer.core.projects import emoca_fit
from AU_recognizer.core.projects.emoca_tag import emoca_tag
from AU_recognizer.core.projects.models.DecaFLAME import FLAME_mediapipe
from AU_recognizer.core.projects.utils.utility import load_model
from AU_recognizer.core.user_interface import CustomFrame, CustomButton, CustomCheckBox, CustomEntry, ThemeManager, \
    CustomTabview, CustomLabel
from AU_recognizer.core.user_interface.dialogs import Dialog
from AU_recognizer.core.user_interface.dialogs.dialog import DialogMessage
from AU_recognizer.core.user_interface.dialogs.dialog_util import open_message_dialog
from AU_recognizer.core.user_interface.widgets.complex_widget import EntryButton, ComboLabel, ColorPickerLabel, \
    NumberPicker
from AU_recognizer.core.util import i18n, PA_NAME, TD_IMAGES, TD_FITTED, TD_HIDE_FITTED, logger, T_CENTER, T_COLUMNS, \
    I18N_CLOSE_BUTTON, I18N_FIT_SEL_BUTTON, I18N_FIT_ALL_BUTTON, P_PATH, F_INPUT, F_OUTPUT, MF_MODEL, I18N_TITLE, \
    I18N_MESSAGE, I18N_DETAIL, INFORMATION_ICON, TD_MESH_N, TD_MESH_C, I18N_COMPARE_SEL_BUTTON, F_COMPARE, OBJ, \
    nect_config, CONFIG, MODEL_FOLDER, GENERAL_TAB, VIEWER_TAB, I18N_PATH, LOGGER_PATH, LOG_FOLDER, PROJECTS_FOLDER, \
    LANGUAGE, retrieve_files_from_path, VIEWER, FILL_COLOR, LINE_COLOR, CANVAS_COLOR, POINT_COLOR, POINT_SIZE, \
    GROUND_COLOR, SKY_COLOR, MOVING_STEP, I18N_SAVE_BUTTON, write_config, MESH_POSE, MESH_IDENTITY, TD_HIDE_IDENTITY, \
    TD_HIDE_POSE, TD_HIDE_NOT_NORM, I18N_TAG_SEL_BUTTON, TD_THRESHOLD
from AU_recognizer.core.util.utility_functions import lighten_color, gray_to_hex


class SelectFitImageDialog(Dialog):
    def __init__(self, master, data, project, title=i18n.project_actions_fit[PA_NAME]):
        super().__init__(master)
        super().title(title)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])
        self.fit_data = data

        self.main_frame = CustomFrame(self)
        self.buttons_frame = CustomFrame(self.main_frame)
        self.bottom_frame = CustomFrame(self.main_frame)
        self.left_frame = CustomFrame(self.main_frame)
        self.right_frame = CustomFrame(self.main_frame)
        # Tabelle delle immagini
        self.available_treeview = Treeview(self.left_frame, columns=(TD_IMAGES, TD_FITTED), selectmode='extended')
        self.selected_treeview = Treeview(self.right_frame, columns=(TD_IMAGES, TD_FITTED), selectmode='extended')
        # between table buttons
        self.move_all_right_button = CustomButton(self.buttons_frame, text=">>", command=self.move_all_to_selected,
                                                  width=30)
        self.move_right_button = CustomButton(self.buttons_frame, text=">", command=self.move_to_selected, width=30)
        self.move_left_button = CustomButton(self.buttons_frame, text="<", command=self.move_to_available, width=30)
        self.move_all_left_button = CustomButton(self.buttons_frame, text="<<", command=self.move_all_to_available,
                                                 width=30)
        # Variable for checkbox state
        self.hide_fitted_var = BooleanVar(value=False)
        # Create the checkbox
        self.hide_fitted_checkbox = CustomCheckBox(
            self.left_frame,
            text=i18n.im_sel_dialog[TD_HIDE_FITTED],
            variable=self.hide_fitted_var,
            command=self.populate_available_treeview  # Refresh the Treeview when toggled
        )
        # Entry widget for filter text
        self.filter_var = StringVar()  # StringVar to hold the filter input
        self.filter_entry = CustomEntry(
            self.left_frame,
            textvariable=self.filter_var,
            validate="key"
        )
        self.filter_entry.bind("<KeyRelease>", self.filter_images)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=True)
        self.bottom_frame.pack(side=BOTTOM, fill=X)
        self.left_frame.pack(side=LEFT, fill=Y, expand=True)
        self.buttons_frame.pack(side=LEFT, padx=10)
        self.right_frame.pack(side=LEFT, fill=BOTH, expand=True)
        # Configurazione Treeview
        for treeview in (self.available_treeview, self.selected_treeview):
            treeview.column(TD_IMAGES, anchor=T_CENTER, minwidth=300, width=300)
            treeview.column(TD_FITTED, anchor=T_CENTER, minwidth=100, width=100, stretch=False)
            treeview.heading(TD_IMAGES, text=i18n.im_sel_dialog[T_COLUMNS][TD_IMAGES])
            treeview.heading(TD_FITTED, text=i18n.im_sel_dialog[T_COLUMNS][TD_FITTED])
            treeview["show"] = "headings"
            # select mode
            treeview["selectmode"] = "extended"
            # displayColumns
            treeview["displaycolumns"] = [TD_IMAGES, TD_FITTED]
        # show
        # Add the checkbox above the Treeview
        self.hide_fitted_checkbox.pack(anchor='w')
        self.available_treeview.pack(fill=BOTH, expand=True)
        self.selected_treeview.pack(side=LEFT, fill=BOTH, expand=True)
        # Bottom frame for buttons
        CustomLabel(master=self.left_frame, text=i18n.mesh_tag_dialog[TD_THRESHOLD]).pack()
        self.filter_entry.pack(fill=X, padx=5, pady=5)  # Filter text box
        # Buttons
        close_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=RIGHT, padx=5, pady=5)
        fit_selected_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_SEL_BUTTON],
                                           command=self.fit_selected)
        fit_selected_button.pack(side=RIGHT, padx=5, pady=5)
        fit_all_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_ALL_BUTTON],
                                      command=self.fit_all)
        fit_all_button.pack(side=RIGHT, padx=5, pady=5)
        # arrows
        self.move_all_right_button.pack(pady=5)
        self.move_right_button.pack(pady=5)
        self.move_left_button.pack(pady=5)
        self.move_all_left_button.pack(pady=5)
        self.available_treeview.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.available_treeview.tag_configure('odd',
                                              background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                  ThemeManager.theme["CustomFrame"]["fg_color"])), 10))
        self.selected_treeview.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.selected_treeview.tag_configure('odd',
                                             background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                 ThemeManager.theme["CustomFrame"]["fg_color"])), 10))
        # Populate the treeview with images
        self.populate_available_treeview()
        self.retag_treeview()

    def filter_images(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_available_treeview()  # Repopulate treeview with filter applied
        self.retag_treeview()

    def populate_available_treeview(self):
        logger.debug(f"{self.__class__.__name__} populate image treeview")
        # Clear existing entries in the Treeview
        self.available_treeview.delete(*self.available_treeview.get_children())
        # List all image files in the folder_path
        images_path = Path(self.project[self._project_name][P_PATH]) / F_INPUT
        # List of desired image suffixes
        image_suffixes = ['.jpg', '.png', '.bmp']
        # Get the filter value from the Entry widget
        filter_text = self.filter_var.get().lower()
        selected_images = {self.selected_treeview.item(item, 'values')[0] for item in
                           self.selected_treeview.get_children()}
        for idx, image in enumerate(sorted(images_path.iterdir(), key=lambda x: x.name.lower())):
            if image.suffix.lower() in image_suffixes and filter_text in image.name.lower():
                if image.name in selected_images:
                    continue
                output_folder = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / self.fit_data[MF_MODEL]
                fit_status = i18n.im_sel_dialog["data"]["fitted"] if output_folder.exists() and (
                    any(d.is_dir() and d.name.startswith(image.stem) for d in output_folder.iterdir())) else \
                    i18n.im_sel_dialog["data"]["not_fitted"]
                if not self.hide_fitted_var.get() or fit_status == i18n.im_sel_dialog["data"]["not_fitted"]:
                    self.available_treeview.insert('', END, values=(image.name, fit_status))

    def fit_all(self):
        logger.debug(f"{self.__class__.__name__} fit all images")
        emoca_fit(master=self, fit_data=self.fit_data,
                  images_to_fit=Path(self.project[self._project_name][P_PATH]) / F_INPUT,
                  project_data=self.project)
        logger.debug("<<UpdateTreeSmall>> event generation")
        self.master.event_generate("<<UpdateTreeSmall>>")

    def fit_selected(self):
        logger.debug(f"{self.__class__.__name__} fit selected images")
        selected_items = self.selected_treeview.get_children()
        images_to_fit = []
        for item in selected_items:
            image_name = self.selected_treeview.item(item, 'values')[0]
            images_to_fit.append(Path(self.project[self._project_name][P_PATH]) / F_INPUT / image_name)
        if images_to_fit:
            emoca_fit(master=self, fit_data=self.fit_data, images_to_fit=images_to_fit, project_data=self.project)
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

    def move_all_to_selected(self):
        """Move all images from available to selected."""
        for item in self.available_treeview.get_children():
            values = self.available_treeview.item(item, 'values')
            if isinstance(values, (list, tuple)):
                self.selected_treeview.insert('', END, values=values)
        self.available_treeview.delete(*self.available_treeview.get_children())
        self.retag_treeview()

    def move_to_selected(self):
        for item in self.available_treeview.selection():
            values = self.available_treeview.item(item, 'values')
            if isinstance(values, (list, tuple)):
                self.selected_treeview.insert('', END, values=values)
            self.available_treeview.delete(item)
        self.retag_treeview()

    def move_all_to_available(self):
        """Move all images from selected back to available."""
        self.selected_treeview.delete(*self.selected_treeview.get_children())
        self.populate_available_treeview()
        self.retag_treeview()

    def move_to_available(self):
        for item in self.selected_treeview.selection():
            self.selected_treeview.delete(item)
            self.populate_available_treeview()
        self.retag_treeview()

    def retag_treeview(self):
        """Reapply 'even' and 'odd' tags after moving items."""
        for idx, item in enumerate(self.selected_treeview.get_children()):
            tag = 'even' if idx % 2 == 0 else 'odd'
            self.selected_treeview.item(item, tags=(tag,))
        for idx, item in enumerate(self.available_treeview.get_children()):
            tag = 'even' if idx % 2 == 0 else 'odd'
            self.available_treeview.item(item, tags=(tag,))


class SelectMeshDialog(Dialog):
    def __init__(self, master, project, title=i18n.project_actions_au_rec[PA_NAME]):
        super().__init__(master)
        self.pose_checkbox = None
        self.identity_checkbox = None
        super().title(title)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])
        self.selected_images = []
        self.normalize_pose_var = BooleanVar(value=True)
        self.normalize_identity_var = BooleanVar(value=False)
        self.main_frame = CustomFrame(self)
        self.neutral_frame = CustomFrame(self.main_frame)
        self.compare_frame = CustomFrame(self.main_frame)
        self.instructions_frame = CustomFrame(self.main_frame)
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
        self.compare_frame.pack(side=LEFT, fill=BOTH, expand=True)
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
        self.instructions_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        # Checkbox for Pose Normalization
        self.pose_checkbox = CustomCheckBox(
            self.instructions_frame,
            text=i18n.mesh_sel_dialog[MESH_POSE],
            variable=self.normalize_pose_var
        )
        self.pose_checkbox.pack(side=TOP, padx=5, pady=5)
        # Checkbox for Identity Normalization
        self.identity_checkbox = CustomCheckBox(
            self.instructions_frame,
            text=i18n.mesh_sel_dialog[MESH_IDENTITY],
            variable=self.normalize_identity_var
        )
        self.identity_checkbox.pack(side=TOP, padx=5, pady=5)
        # Bottom frame for buttons
        self.bottom_frame.pack(side=BOTTOM, fill=X, expand=True)
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
        open_message_dialog(master=self, message="compare_success")
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
        neutral_obj = OBJ(filepath=neutral_face_path / "mesh_coarse.obj", generate_normals=False)
        n_verts = neutral_obj.get_vertices()
        neutral_pose = np.load(neutral_face_path / 'pose.npy')
        neutral_shape = np.load(neutral_face_path / 'shape.npy')
        # load emoca model
        path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER])
        model_name = "EMOCA_v2_lr_mse_20"
        mode = "detail"
        # Build a suffix string based on normalization options
        suffix = ""
        if self.normalize_pose_var.get():
            suffix += "_normPose"
        if self.normalize_identity_var.get():
            suffix += "_normIdentity"
        # 0) clear memory
        torch.cuda.empty_cache()  # Clear any cached GPU memory
        # 1) Load the model
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()
        for face_path in compare_list_paths:
            face_path = Path(face_path)
            face_obj = OBJ(filepath=face_path / "mesh_coarse.obj", generate_normals=False)
            f_exp = np.load(face_path / 'exp.npy')
            f_pose = np.load(face_path / 'pose.npy')
            f_shape = np.load(face_path / 'shape.npy')
            shape = torch.from_numpy(neutral_shape if self.normalize_identity_var.get() else f_shape)
            exp = torch.from_numpy(f_exp)
            pose = torch.from_numpy(neutral_pose if self.normalize_pose_var.get() else f_pose)
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
            compare_obj = copy.deepcopy(face_obj)
            compare_obj.set_vertex_colors(vertex_colors)
            compare_obj.save(output_path / f"{face_path.name[:-2]}_with_heatmap{suffix}.obj")
            # save diff
            # Usa una regex per inserire un "_" tra lettere e numeri
            modified_name = re.sub(r'([A-Za-z]+)(\d)', r'\1_\2', face_path.name[:-2])
            np.save(output_path / f"{modified_name}_diffs{suffix}.npy", diffs)  # Save as binary
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
        neutral_face_name = neutral_face_path.name[:-2]
        config[neutral_face_name] = {}
        # Log the neutral face's comparison to each face in compare_list_paths
        for i, compare_face_path in enumerate(compare_list_paths):
            compare_face_name = Path(compare_face_path).name[:-2]
            config[neutral_face_name][f"compared_face_{i + 1}"] = str(compare_face_name)
        # Save the log to the .ini file
        with open(log_file, 'w') as log:
            config.write(log)
        logger.debug(f"Comparison log saved to {log_file}")


def validate_threshold(new_value):
    """Validation function to ensure input is a float between 0 and 1."""
    if new_value == "":  # Allow empty input while typing
        return True
    try:
        value = float(new_value)
        return 0.0 <= value <= 1.0  # Allow only values between 0 and 1
    except ValueError:
        return False  # Reject invalid input


class TagMeshDialog(Dialog):
    def __init__(self, master, project, title=i18n.project_actions_fit[PA_NAME]):
        super().__init__(master)
        super().title(title)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])

        self.main_frame = CustomFrame(self)
        self.buttons_frame = CustomFrame(self.main_frame)
        self.bottom_frame = CustomFrame(self.main_frame)
        self.left_frame = CustomFrame(self.main_frame)
        self.right_frame = CustomFrame(self.main_frame)
        # Tabelle delle immagini
        self.differences_treeview = Treeview(self.left_frame, columns=TD_IMAGES, selectmode='extended')
        self.selected_treeview = Treeview(self.right_frame, columns=TD_IMAGES, selectmode='extended')
        # between table buttons
        self.move_all_right_button = CustomButton(self.buttons_frame, text=">>", command=self.move_all_to_selected,
                                                  width=30)
        self.move_right_button = CustomButton(self.buttons_frame, text=">", command=self.move_to_selected, width=30)
        self.move_left_button = CustomButton(self.buttons_frame, text="<", command=self.move_to_available, width=30)
        self.move_all_left_button = CustomButton(self.buttons_frame, text="<<", command=self.move_all_to_available,
                                                 width=30)
        # Variable for checkbox state
        self.hide_not_norm = BooleanVar(value=False)
        self.hide_norm_pose = BooleanVar(value=False)
        self.hide_norm_identity = BooleanVar(value=False)
        # Create the checkbox
        self.hide_not_norm_checkbox = CustomCheckBox(
            self.left_frame,
            text=i18n.mesh_tag_dialog[TD_HIDE_NOT_NORM],
            variable=self.hide_not_norm,
            command=lambda: self.filter_images(None)  # Refresh the Treeview when toggled
        )
        self.hide_pose_checkbox = CustomCheckBox(
            self.left_frame,
            text=i18n.mesh_tag_dialog[TD_HIDE_POSE],
            variable=self.hide_norm_pose,
            command=lambda: self.filter_images(None)  # Refresh the Treeview when toggled
        )
        self.hide_identity_checkbox = CustomCheckBox(
            self.left_frame,
            text=i18n.mesh_tag_dialog[TD_HIDE_IDENTITY],
            variable=self.hide_norm_identity,
            command=lambda: self.filter_images(None)  # Refresh the Treeview when toggled
        )
        # Entry widget for filter text
        self.filter_var = StringVar()  # StringVar to hold the filter input
        self.filter_entry = CustomEntry(
            self.left_frame,
            textvariable=self.filter_var,
            validate="key"
        )
        self.filter_entry.bind("<KeyRelease>", self.filter_images)
        self.threshold_var = StringVar(value="0.2")  # StringVar to hold the filter input
        self.threshold_entry = CustomEntry(
            self.right_frame,
            textvariable=self.threshold_var,
            validate="key",
            validatecommand=(self.register(validate_threshold), "%P")
        )

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=True)
        self.bottom_frame.pack(side=BOTTOM, fill=X)
        self.left_frame.pack(side=LEFT, fill=Y, expand=True)
        self.buttons_frame.pack(side=LEFT, padx=10)
        self.right_frame.pack(side=LEFT, fill=BOTH, expand=True)
        # Configurazione Treeview
        for treeview in (self.differences_treeview, self.selected_treeview):
            treeview.column(TD_IMAGES, anchor=T_CENTER, minwidth=300, width=300)
            treeview.heading(TD_IMAGES, text=i18n.mesh_tag_dialog[T_COLUMNS][TD_IMAGES])
            treeview["show"] = "headings"
            # select mode
            treeview["selectmode"] = "extended"
            # displayColumns
            treeview["displaycolumns"] = [TD_IMAGES]
        # show
        # Add the checkbox above the Treeview
        self.hide_not_norm_checkbox.pack(anchor='w')
        self.hide_pose_checkbox.pack(anchor='w')
        self.hide_identity_checkbox.pack(anchor='w')
        self.differences_treeview.pack(fill=BOTH, expand=True)
        self.selected_treeview.pack(side=LEFT, fill=BOTH, expand=True)
        # Bottom frame for buttons
        self.filter_entry.pack(fill=X, padx=5, pady=5)  # Filter text box
        self.threshold_entry.pack(fill=X, padx=5, pady=5)  # Filter text box
        # Buttons
        close_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=RIGHT, padx=5, pady=5)
        tag_selected_button = CustomButton(self.bottom_frame, text=i18n.dialog_buttons[I18N_TAG_SEL_BUTTON],
                                           command=self.tag_selected)
        tag_selected_button.pack(side=RIGHT, padx=5, pady=5)
        # arrows
        self.move_all_right_button.pack(pady=5)
        self.move_right_button.pack(pady=5)
        self.move_left_button.pack(pady=5)
        self.move_all_left_button.pack(pady=5)
        self.differences_treeview.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.differences_treeview.tag_configure('odd',
                                                background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                    ThemeManager.theme["CustomFrame"]["fg_color"])), 10))
        self.selected_treeview.tag_configure('even', background=self._apply_appearance_mode(
            ThemeManager.theme["CustomFrame"]["fg_color"]))
        self.selected_treeview.tag_configure('odd',
                                             background=lighten_color(gray_to_hex(self._apply_appearance_mode(
                                                 ThemeManager.theme["CustomFrame"]["fg_color"])), 10))
        # Populate the treeview with images
        self.populate_available_treeview()
        self.retag_treeview()

    def filter_images(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_available_treeview()  # Repopulate treeview with filter applied
        self.retag_treeview()

    def populate_available_treeview(self):
        logger.debug(f"{self.__class__.__name__} populate image treeview")
        # Clear existing entries in the Treeview
        self.differences_treeview.delete(*self.differences_treeview.get_children())
        # List all image files in the folder_path
        differences_path = Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / F_COMPARE
        # List of desired image suffixes
        differences_suffixes = ['.npy']
        hide_not_norm = self.hide_not_norm.get()
        hide_pose = self.hide_norm_pose.get()
        hide_identity = self.hide_norm_identity.get()
        # Get the filter value from the Entry widget
        filter_text = self.filter_var.get().lower()
        selected_differences = {self.selected_treeview.item(item, 'values')[0] for item in
                                self.selected_treeview.get_children()}
        for idx, diff in enumerate(sorted(differences_path.iterdir(), key=lambda x: x.name.lower())):
            if diff.suffix.lower() in differences_suffixes and filter_text in diff.name.lower():
                # Apply filters
                if diff.name in selected_differences:
                    continue
                if hide_not_norm and "norm" not in diff.name.lower():
                    continue
                if hide_pose and "pose" in diff.name.lower():
                    continue
                if hide_identity and "identity" in diff.name.lower():
                    continue
                self.differences_treeview.insert('', END, values=(diff.name,))

    def tag_selected(self):
        logger.debug(f"{self.__class__.__name__} tag selected differences")
        selected_items = self.selected_treeview.get_children()
        diff_to_tag = []
        for item in selected_items:
            diff_name = self.selected_treeview.item(item, 'values')[0]
            diff_to_tag.append(Path(self.project[self._project_name][P_PATH]) / F_OUTPUT / F_COMPARE / diff_name)
        if diff_to_tag:
            emoca_tag(diff_to_tag=diff_to_tag, threshold=float(self.threshold_var.get())
                      , project_data=self.project)
        else:
            logger.info("No diff  where selected.")
            data = i18n.project_message["no_diff_selected"]
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

    def move_all_to_selected(self):
        """Move all images from available to selected."""
        for item in self.differences_treeview.get_children():
            values = self.differences_treeview.item(item, 'values')
            if isinstance(values, (list, tuple)):
                self.selected_treeview.insert('', END, values=values)
        self.differences_treeview.delete(*self.differences_treeview.get_children())
        self.retag_treeview()

    def move_to_selected(self):
        for item in self.differences_treeview.selection():
            values = self.differences_treeview.item(item, 'values')
            if isinstance(values, (list, tuple)):
                self.selected_treeview.insert('', END, values=values)
            self.differences_treeview.delete(item)
        self.retag_treeview()

    def move_all_to_available(self):
        """Move all images from selected back to available."""
        self.selected_treeview.delete(*self.selected_treeview.get_children())
        self.populate_available_treeview()
        self.retag_treeview()

    def move_to_available(self):
        for item in self.selected_treeview.selection():
            self.selected_treeview.delete(item)
            self.populate_available_treeview()
        self.retag_treeview()

    def retag_treeview(self):
        """Reapply 'even' and 'odd' tags after moving items."""
        for idx, item in enumerate(self.selected_treeview.get_children()):
            tag = 'even' if idx % 2 == 0 else 'odd'
            self.selected_treeview.item(item, tags=(tag,))
        for idx, item in enumerate(self.differences_treeview.get_children()):
            tag = 'even' if idx % 2 == 0 else 'odd'
            self.differences_treeview.item(item, tags=(tag,))


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
