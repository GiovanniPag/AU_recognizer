import tkinter as tk
from tkinter import ttk
from abc import abstractmethod

from AU_recognizer.core.projects.emoca_fitter import emoca_fit
from AU_recognizer.core.util import retrieve_files_from_path
from AU_recognizer.core.util.config import logger, nect_config, write_config
from AU_recognizer.core.util.constants import *
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.views import FloatPicker
from AU_recognizer.core.views import IntPicker
from AU_recognizer.core.views import ColorPicker
from AU_recognizer.core.views import ComboLabel
from AU_recognizer.core.views import EntryButton
from AU_recognizer.core.util.utility_functions import check_name, check_file_name
from AU_recognizer.core.views import ScrollWrapperView


class Dialog(tk.Toplevel):
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
        self.name_var = tk.StringVar()
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
        tk.Label(self, image=FULL_QUESTION_ICON).grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7), sticky="e")
        ttk.Label(self, text=self.name_label, font="bold").grid(row=0, column=1, columnspan=1, pady=(7, 7), padx=(7, 7),
                                                                sticky="w")
        entry = ttk.Entry(self, textvariable=self.name_var, validate='key',
                          validatecommand=(self.master.register(self.validate_command), '%P'))
        entry.bind("<Return>", lambda event: self.close())
        entry.grid(row=0, column=2, columnspan=3, pady=(7, 7), padx=(7, 7), sticky="we")
        ttk.Label(self, text=self.name_tip, background="red", wraplength=400).grid(row=1, column=1, columnspan=4,
                                                                                   padx=(7, 7), sticky="we")
        ttk.Button(self, text=i18n.dialog_buttons[I18N_CONFIRM_BUTTON], command=self.close).grid(row=2, column=2,
                                                                                                 pady=(7, 7),
                                                                                                 padx=(7, 7),
                                                                                                 sticky="e")
        c_c = 3
        if self.has_back:
            c_c = 4
            ttk.Button(self, text=i18n.dialog_buttons[I18N_BACK_BUTTON], command=self.go_back).grid(row=2, column=3,
                                                                                                    pady=(7, 7),
                                                                                                    padx=(7, 7),
                                                                                                    sticky="e")
        ttk.Button(self, text=i18n.dialog_buttons[I18N_CANCEL_BUTTON], command=self.dismiss).grid(row=2, column=c_c,
                                                                                                  pady=(7, 7),
                                                                                                  padx=(7, 7),
                                                                                                  sticky="e")
        entry.focus_set()

    def go_back(self):
        if self.has_back:
            logger.debug("Project Options Dialog Back method")
            self.back = True
            self.close()


# TODO: show fitting process, and return message, also check error on coarse fitting, return nulls and crashes
class SelectFitImageDialog(Dialog):
    def __init__(self, master, data, project):
        super().__init__(master)
        self.master = master
        self.project = project
        self._project_name = str(self.project.sections()[0])
        self.selected_images = []
        self.fit_data = data
        self.main_frame = tk.Frame(self, padx=20, pady=20)
        self.scroll_frame = ScrollWrapperView(master=self.main_frame)
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.image_treeview = ttk.Treeview(self.scroll_frame, selectmode='extended')
        # Variable for checkbox state
        self.hide_fitted_var = tk.BooleanVar(value=False)
        # Create the checkbox
        self.hide_fitted_checkbox = ttk.Checkbutton(
            self.main_frame,
            text=i18n.im_sel_dialog[TD_HIDE_FITTED],
            variable=self.hide_fitted_var,
            command=self.populate_image_treeview  # Refresh the Treeview when toggled
        )
        # Entry widget for filter text
        self.filter_var = tk.StringVar()  # StringVar to hold the filter input
        self.filter_entry = ttk.Entry(
            self.main_frame,
            textvariable=self.filter_var,
            validate="key"
        )
        self.filter_entry.bind("<KeyRelease>", self.filter_images)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view")
        # Main frame
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.scroll_frame.add(self.image_treeview)
        self.filter_entry.pack(fill=tk.X, padx=5, pady=5)  # Filter text box
        # Add the checkbox above the Treeview
        self.hide_fitted_checkbox.pack(anchor='w')
        # Treeview for images
        # Treeview for images
        self.image_treeview[T_COLUMNS] = (TD_IMAGES, TD_FITTED)
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
        # Bottom frame for buttons
        self.bottom_frame.pack(fill=tk.X, expand=True)
        # Buttons
        close_button = ttk.Button(self.bottom_frame, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)
        fit_selected_button = ttk.Button(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_SEL_BUTTON],
                                         command=self.fit_selected)
        fit_selected_button.pack(side=tk.RIGHT, padx=5, pady=5)
        fit_all_button = ttk.Button(self.bottom_frame, text=i18n.dialog_buttons[I18N_FIT_ALL_BUTTON],
                                    command=self.fit_all)
        fit_all_button.pack(side=tk.RIGHT, padx=5, pady=5)
        # Populate the treeview with images
        self.populate_image_treeview()

    def filter_images(self, event):
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
                self.image_treeview.insert('', tk.END, values=(image.name, fit_status, image), tags=(tag,))
        # Define tags and styles for alternating row colors
        self.image_treeview.tag_configure('even', background='#f0f0ff')
        self.image_treeview.tag_configure('odd', background='#ffffff')

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


class SettingsDialog(Dialog):
    def __init__(self, master, page=""):
        super().__init__(master)
        self.master = master
        # Create the notebook
        self.notebook = ttk.Notebook(self)

        # Create frames for the tabs
        self.general_frame = ttk.Frame(self.notebook)
        self.viewer_frame = ttk.Frame(self.notebook)
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
        self._fill_color = ColorPicker(master=self.viewer_frame, label_text="fill_color",
                                       default=nect_config[VIEWER][FILL_COLOR])
        self._line_color = ColorPicker(master=self.viewer_frame, label_text="line_color",
                                       default=nect_config[VIEWER][LINE_COLOR])
        self._canvas_color = ColorPicker(master=self.viewer_frame, label_text="canvas_color",
                                         default=nect_config[VIEWER][CANVAS_COLOR])
        self._point_color = ColorPicker(master=self.viewer_frame, label_text="point_color",
                                        default=nect_config[VIEWER][POINT_COLOR])
        self._point_size = IntPicker(master=self.viewer_frame, label_text="point_size",
                                     default=nect_config[VIEWER][POINT_SIZE], min_value=1, max_value=20)
        self._ground_color = ColorPicker(master=self.viewer_frame, label_text="ground_color",
                                         default=nect_config[VIEWER][GROUND_COLOR])
        self._sky_color = ColorPicker(master=self.viewer_frame, label_text="sky_color",
                                      default=nect_config[VIEWER][SKY_COLOR])
        self._moving_step = FloatPicker(master=self.viewer_frame, label_text="moving_step",
                                        default=nect_config[VIEWER][MOVING_STEP], min_value=0.01, max_value=1,
                                        increment=0.01)

    def create_view(self):
        logger.debug(f"{self.__class__.__name__} create view method")
        self.grid_columnconfigure(0, weight=1)
        self.notebook.grid(row=0, column=0, sticky=tk.NSEW, columnspan=2, padx=10)
        # Add frames to notebook as tabs
        self.__create_general_tab()
        self.__create_viewer_tab()
        save_button = tk.Button(self, text=i18n.dialog_buttons[I18N_SAVE_BUTTON], command=self.save_config)
        save_button.grid(row=1, column=0, sticky=tk.E, padx=10, pady=10)
        close_button = tk.Button(self, text=i18n.dialog_buttons[I18N_CLOSE_BUTTON], command=self.close)
        close_button.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=10)
        if self.startpage == "viewer":
            self.notebook.select(self.viewer_frame)

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

    def ask_value(self):
        logger.debug(f"{self.__class__.__name__} ask value")
        self.master.event_generate("<<ViewerChange>>")

    def dismiss_method(self):
        logger.debug(f"{self.__class__.__name__} dismiss method")
        pass

    def __create_general_tab(self):
        self.notebook.add(self.general_frame, text=i18n.settings_dialog[GENERAL_TAB])
        self.general_frame.grid_columnconfigure(0, weight=1)
        self._i18n_path.grid(row=0, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self._i18n_path.create_view()
        self.languages_combobox.grid(row=1, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self.languages_combobox.create_view()
        self._logger_path.grid(row=2, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self._logger_path.create_view()
        self._log_folder.grid(row=3, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self._log_folder.create_view()
        self._projects_folder.grid(row=4, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self._projects_folder.create_view()
        self._path_to_model.grid(row=5, column=0, sticky=tk.EW, columnspan=2, padx=10)
        self._path_to_model.create_view()

    def __create_viewer_tab(self):
        self.notebook.add(self.viewer_frame, text=i18n.settings_dialog[VIEWER_TAB])
        self.viewer_frame.grid_columnconfigure(0, weight=1)
        self._fill_color.grid(row=0, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._fill_color.create_view()
        self._line_color.grid(row=1, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._line_color.create_view()
        self._canvas_color.grid(row=2, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._canvas_color.create_view()
        self._ground_color.grid(row=4, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._ground_color.create_view()
        self._sky_color.grid(row=5, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._sky_color.create_view()
        self._point_color.grid(row=3, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._point_color.create_view()
        self._point_size.grid(row=6, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
        self._point_size.create_view()
        self._moving_step.grid(row=7, column=0, sticky=tk.EW, columnspan=2, padx=10, pady=5)
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
        tk.Label(self, image=BASENAME_ICON + self.icon) \
            .grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7), sticky="e")
        ttk.Label(self, text=self.message, font="bold") \
            .grid(row=0, column=1, columnspan=2, pady=(7, 7), padx=(7, 7), sticky="w")
        ttk.Label(self, text=self.detail).grid(row=1, column=1, columnspan=2, pady=(7, 7), padx=(7, 7), sticky="w")
        ttk.Button(self, text=i18n.dialog_buttons[I18N_BACK_BUTTON], command=self.dismiss) \
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
        self.response_var = tk.StringVar(value=dismiss_response)

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
        tk.Label(self, image=BASENAME_ICON + self.icon) \
            .grid(row=0, column=0, rowspan=2, pady=(7, 0), padx=(7, 7), sticky="e")
        ttk.Label(self, text=self.message, font="bold") \
            .grid(row=0, column=1, columnspan=(1 + len(self.options)), pady=(7, 7), padx=(7, 7), sticky="w")
        ttk.Label(self, text=self.detail).grid(row=1, column=1, columnspan=(1 + len(self.options)), pady=(7, 7),
                                               padx=(7, 7), sticky="w")
        for index, option in enumerate(self.options):
            b = ttk.Button(self, text=i18n.dialog_buttons[option], command=lambda opt=option: self.response(opt))
            b.grid(row=2, column=2 + index, pady=(7, 7), padx=(7, 7), sticky="e")
