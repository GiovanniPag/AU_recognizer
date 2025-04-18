import configparser
from configparser import ConfigParser
from pathlib import Path
from tkinter import StringVar, BooleanVar, END, BOTH, LEFT, X, RIGHT, TOP, BOTTOM
from tkinter.ttk import Treeview
from typing import Optional

from AU_recognizer.core.model.model_manager import load_model_class
from AU_recognizer.core.user_interface import CustomButton, CustomFrame, CustomEntry, ThemeManager, CustomCheckBox
from AU_recognizer.core.user_interface.dialogs.dialog import DialogMessage
from AU_recognizer.core.user_interface.dialogs.dialog_util import open_message_dialog
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import logger, i18n, AU_SELECT_MESH, AU_TAG_MESH, TD_MESH_N, TD_MESH_C, P_PATH, F_OUTPUT, \
    T_CENTER, T_COLUMNS, MESH_POSE, MESH_IDENTITY, I18N_COMPARE_SEL_BUTTON, I18N_TITLE, I18N_MESSAGE, I18N_DETAIL, \
    INFORMATION_ICON, nect_config, CONFIG, MODEL, F_COMPARE
from AU_recognizer.core.util.utility_functions import lighten_color, gray_to_hex


class AURecognitionView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.model = nect_config[CONFIG][MODEL]
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._project_name = None
        self.selected_images = []
        self.directories_useful = []
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
        self.pose_checkbox = None
        self.identity_checkbox = None

        self._au_button_label = StringVar()
        self.identity_checkbox_label = StringVar()
        self.pose_checkbox_label = StringVar()
        self.au_button: Optional[CustomButton] = None
        self.update_language()

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
            textvariable=self.pose_checkbox_label,
            variable=self.normalize_pose_var
        )
        self.pose_checkbox.pack(side=TOP, padx=5, pady=5)
        # Checkbox for Identity Normalization
        self.identity_checkbox = CustomCheckBox(
            self.instructions_frame,
            textvariable=self.identity_checkbox_label,
            variable=self.normalize_identity_var
        )
        self.identity_checkbox.pack(side=TOP, padx=5, pady=5)
        # Bottom frame for buttons
        self.bottom_frame.pack(side=BOTTOM, fill=X, expand=True)
        # Buttons
        self.au_button = CustomButton(self.bottom_frame, textvariable=self._au_button_label,
                                      command=self.compare_selected)
        self.au_button.pack(side=RIGHT, padx=5, pady=5)

    def update_language(self):
        logger.debug("update language in AURecognition view")
        self._au_button_label.set(i18n.dialog_buttons[I18N_COMPARE_SEL_BUTTON])
        self.identity_checkbox_label.set(i18n.mesh_sel_dialog[MESH_IDENTITY])
        self.pose_checkbox_label.set(i18n.mesh_sel_dialog[MESH_POSE])
        self.mesh_treeview_compare.heading(TD_MESH_C, text=i18n.mesh_sel_dialog[T_COLUMNS][TD_MESH_C])
        self.mesh_treeview_neutral.heading(TD_MESH_N, text=i18n.mesh_sel_dialog[T_COLUMNS][TD_MESH_N])

    def update_selected_project(self, data=None):
        logger.debug("update selected project in AURecognition view")
        self._project_info = data
        self._project_name = str(self._project_info.sections()[0])
        # List all image files in the folder_path
        models_path = Path(self._project_info[self._project_name][P_PATH]) / F_OUTPUT
        # List of required files
        required_files = ['mesh_coarse.obj', 'pose.npy', 'shape.npy', 'tex.npy', 'detail.npy', 'exp.npy']
        # List all directories inside mesh_path and check for the required files
        self.directories_useful = []
        for directory in models_path.rglob('*'):  # rglob searches recursively:
            if directory.is_dir() and self.contains_required_files(directory, required_files):
                self.directories_useful.append(directory)
        # Populate the treeview with images
        self.populate_neutral_treeview()
        self.populate_compare_treeview()

    # Function to check if all required files are in a directory
    @staticmethod
    def contains_required_files(directory, required_files):
        return all((directory / file).exists() for file in required_files)

    def filter_mesh_neutral(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_neutral_treeview()  # Repopulate treeview with filter applied

    def filter_mesh_compare(self, _event):
        """This method will be triggered every time the user types in the filter Entry widget."""
        self.populate_compare_treeview()  # Repopulate treeview with filter applied

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

    def compare_faces(self, neutral_face_path, compare_list_paths):
        logger.debug(f"compare {neutral_face_path} with list {compare_list_paths}")
        model_class = load_model_class(self.model)
        model_class.au_difference(mesh_neutral=neutral_face_path, mesh_list=compare_list_paths, normalization_params={
            "pose": self.normalize_pose_var.get(),
            "identity": self.normalize_identity_var.get()
        }, project=self._project_info)
        self.log_face_comparisons(neutral_face_path, compare_list_paths)

    def log_face_comparisons(self, neutral_face_path, compare_list_paths):
        logger.debug(f"Logging comparison between {neutral_face_path} and {compare_list_paths}")
        output_path = Path(self._project_info[self._project_name][P_PATH]) / F_OUTPUT / F_COMPARE
        output_path.mkdir(parents=True, exist_ok=True)
        # Set up the log file path
        log_file = output_path / "face_comparisons.ini"
        # Create configparser object to write to ini file
        config = configparser.ConfigParser()
        # Load the existing file if it exists, to avoid overwriting previous logs
        if log_file.exists():
            config.read(log_file)
        # Create a new section for the neutral face comparison
        neutral_face_name = Path(neutral_face_path).name[:-2]
        config[neutral_face_name] = {}
        # Log the neutral face's comparison to each face in compare_list_paths
        for i, compare_face_path in enumerate(compare_list_paths):
            compare_face_name = Path(compare_face_path).name[:-2]
            config[neutral_face_name][f"compared_face_{i + 1}"] = str(compare_face_name)
        # Save the log to the .ini file
        with open(log_file, 'w') as log:
            config.write(log)
        logger.debug(f"Comparison log saved to {log_file}")

    def update_model(self, new_model=nect_config[CONFIG][MODEL]):
        logger.debug("model changed, update views")
        if self.model != new_model:
            self.model = new_model

