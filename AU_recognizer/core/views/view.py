import tkinter as tk
from configparser import ConfigParser

from tkinter import ttk, VERTICAL
from typing import Optional

from AU_recognizer.core.controllers import Controller
from AU_recognizer.core.util import config
from AU_recognizer.core.util.config import logger, nect_config
from AU_recognizer.core.util.constants import *
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.views import View, AutoScrollbar, ScrollFrame, ComboLabel, CheckLabel, RadioList
from AU_recognizer.core.views.image_viewer import CanvasImage
from AU_recognizer.core.views.viewer_3d import Viewer3D


class ScrollWrapperView(View):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.widget = None
        self.scroll = None

    def add(self, widget):
        logger.debug(f"add widget {widget} to scroll wrapper view")
        self.widget = widget
        self.scroll = AutoScrollbar(self, orient=VERTICAL, command=self.widget.yview, column_grid=1, row_grid=0)
        self.widget.configure(yscrollcommand=self.scroll.set)
        self.widget.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

    def create_view(self):
        logger.debug(f"create view in scroll wrapper view")
        self.widget.create_view()

    def update_language(self):
        logger.debug(f"update language in scroll wrapper view")
        self.widget.update_language()


class MenuBar(tk.Menu, View):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.menu_apple = tk.Menu(self, name=M_APPLE)
        self.menu_file = tk.Menu(self)
        self.menu_help = tk.Menu(self, name=M_HELP)
        self.menu_help_language = tk.Menu(self.menu_help)
        self.language = tk.StringVar()

        self.__menu_names = {
            M_APPLE: {
                M_MASTER: self,
            },
            M_FILE: {
                M_MASTER: self,
                M_INDEX: 0
            },
            M_EDIT: {
                M_MASTER: self,
                M_INDEX: 1
            },
            M_HELP: {
                M_MASTER: self,
                M_INDEX: 2
            },
            M_NEW: {
                M_MASTER: self.menu_file,
                M_INDEX: 0
            },
            M_OPEN: {
                M_MASTER: self.menu_file,
                M_INDEX: 1
            },
            M_SETTINGS: {
                M_MASTER: self.menu_file,
                M_INDEX: 2
            },
            M_EXIT: {
                M_MASTER: self.menu_file,
                M_INDEX: 3
            },
            M_LANGUAGE: {
                M_MASTER: self.menu_help,
                M_INDEX: 0
            },
            M_ABOUT: {
                M_MASTER: self.menu_help,
                M_INDEX: 2
            },
            M_LOGS: {
                M_MASTER: self.menu_help,
                M_INDEX: 3
            },
            M_GUIDE: {
                M_MASTER: self.menu_help,
                M_INDEX: 4
            },
            M_IT: {
                M_MASTER: self.menu_help_language,
                M_RADIO: True,
                M_VARIABLE: self.language,
                M_VALUE: M_IT,
                M_INDEX: 0
            },
            M_EN: {
                M_MASTER: self.menu_help_language,
                M_RADIO: True,
                M_VARIABLE: self.language,
                M_VALUE: M_EN,
                M_INDEX: 1
            },
        }

    def create_view(self):
        logger.debug(f"create view in menu bar")
        if config.windowing_system == "aqua":
            self.add_cascade_item(menu_name=M_APPLE, menu_to_add=self.menu_apple, info={})
            self.master.createcommand(MAC_SHOW_HELP, ...)
        self.add_cascade_item(menu_name=M_FILE, menu_to_add=self.menu_file, info=i18n.menu_file)
        self.add_cascade_item(menu_name=M_HELP, menu_to_add=self.menu_help, info=i18n.menu_help)
        # menu_file items
        self.add_command_item(cmd_name=M_NEW, info=i18n.menu_file_new)
        self.add_command_item(cmd_name=M_OPEN, info=i18n.menu_file_open)
        self.add_command_item(cmd_name=M_SETTINGS, info=i18n.menu_file_settings)
        self.add_command_item(cmd_name=M_EXIT, info=i18n.menu_file_exit)
        # menu_help items
        self.add_cascade_item(menu_name=M_LANGUAGE, menu_to_add=self.menu_help_language, info=i18n.menu_help_language)
        # separator
        self.menu_help.add_separator()
        self.add_command_item(cmd_name=M_ABOUT, info=i18n.menu_help_about)
        self.add_command_item(cmd_name=M_LOGS, info=i18n.menu_help_logs)
        self.add_command_item(cmd_name=M_GUIDE, info=i18n.menu_help_guide)
        # menu_help_language
        self.add_command_item(cmd_name=M_IT, info=i18n.menu_help_language_it)
        self.add_command_item(cmd_name=M_EN, info=i18n.menu_help_language_en)

    def update_language(self):
        logger.debug(f"{self.winfo_name()} update language")
        self.update_command_or_cascade(name=M_FILE, info_updated=i18n.menu_file)
        self.update_command_or_cascade(name=M_HELP, info_updated=i18n.menu_help)
        # menu_file items
        self.update_command_or_cascade(name=M_NEW, info_updated=i18n.menu_file_new)
        self.update_command_or_cascade(name=M_OPEN, info_updated=i18n.menu_file_open)
        self.update_command_or_cascade(name=M_SETTINGS, info_updated=i18n.menu_file_settings)
        self.update_command_or_cascade(name=M_EXIT, info_updated=i18n.menu_file_exit)
        # menu_help items
        self.update_command_or_cascade(name=M_LANGUAGE, info_updated=i18n.menu_help_language)
        self.update_command_or_cascade(name=M_ABOUT, info_updated=i18n.menu_help_about)
        self.update_command_or_cascade(name=M_LOGS, info_updated=i18n.menu_help_logs)
        self.update_command_or_cascade(name=M_GUIDE, info_updated=i18n.menu_help_guide)
        # menu_help_language items
        self.update_command_or_cascade(name=M_IT, info_updated=i18n.menu_help_language_it)
        self.update_command_or_cascade(name=M_EN, info_updated=i18n.menu_help_language_en)

    def add_cascade_item(self, menu_name, menu_to_add, info):
        logger.debug(f"add cascade item {menu_to_add} to {menu_name} with info {info}")
        self.__menu_names[menu_name][M_MASTER].add_cascade(menu=menu_to_add, label=info.get(M_LABEL, ""),
                                                           underline=info.get(M_UNDERLINE, -1),
                                                           state=info.get(M_DEFAULT_STATE, M_STATE_NORMAL))

    def add_command_item(self, cmd_name, info=None):
        if info:
            logger.debug(f"add command item {cmd_name} with info {info}")
            if self.__menu_names[cmd_name].get(M_RADIO, False):
                self.__menu_names[cmd_name][M_MASTER] \
                    .add_radiobutton(label=info.get(M_LABEL, ""), underline=info.get(M_UNDERLINE, -1),
                                     state=info.get(M_DEFAULT_STATE, M_STATE_NORMAL),
                                     variable=self.__menu_names[cmd_name][M_VARIABLE],
                                     value=self.__menu_names[cmd_name][M_VALUE],
                                     accelerator=info.get(M_ACCELERATOR, ""), command=...)
            else:
                self.__menu_names[cmd_name][M_MASTER] \
                    .add_command(label=info.get(M_LABEL, ""), underline=info.get(M_UNDERLINE, -1),
                                 state=info.get(M_DEFAULT_STATE, M_STATE_NORMAL),
                                 accelerator=info.get(M_ACCELERATOR, ""), command=...)
        else:
            self.__menu_names[cmd_name][M_MASTER] \
                .add_command(label=cmd_name, state=M_STATE_NORMAL, command=...)

    def update_command_or_cascade(self, name, info_updated, update_state=False):
        logger.debug(f"update item {name} with info {info_updated}")
        master = self.__menu_names[name][M_MASTER]
        entry_to_update = self.__menu_names[name][M_INDEX]
        if M_LABEL in info_updated:
            master.entryconfigure(entry_to_update, label=info_updated[M_LABEL])
        if M_UNDERLINE in info_updated:
            master.entryconfigure(entry_to_update, underline=info_updated[M_UNDERLINE])
        if M_STATE in info_updated and update_state:
            master.entryconfigure(entry_to_update, state=info_updated[M_STATE])
        if M_ACCELERATOR in info_updated:
            master.entryconfigure(entry_to_update, accelerator=info_updated[M_ACCELERATOR])
        if M_COMMAND in info_updated:
            master.entryconfigure(entry_to_update, command=info_updated[M_COMMAND])


class TreeViewMenu(tk.Menu, View):

    def __init__(self, master=None):
        super().__init__(master, tearoff=0)
        self.master = master
        self.data = None
        self.is_open = False
        self.__menu_names = {
            MT_SEL_P: {
                M_MASTER: self,
                M_INDEX: 0
            },
            MT_ADD_IMAGES: {
                M_MASTER: self,
                M_INDEX: 1
            },
            # separator here index 2
            MT_SELECT_P: {
                M_MASTER: self,
                M_INDEX: 3
            },
            MT_CLOSE_P: {
                M_MASTER: self,
                M_INDEX: 4
            },
            MT_DELETE_P: {
                M_MASTER: self,
                M_INDEX: 5
            },
            # separator here index 6
            MT_SEL_F: {
                M_MASTER: self,
                M_INDEX: 7
            },
            MT_OPEN_F: {
                M_MASTER: self,
                M_INDEX: 8
            },
            MT_RENAME_F: {
                M_MASTER: self,
                M_INDEX: 9
            },
            MT_DELETE_F: {
                M_MASTER: self,
                M_INDEX: 10
            },
        }

    def create_view(self):
        logger.debug(f"create view in context menu bar")
        # selected project
        self.add_command_item(cmd_name=MT_SEL_P, info=i18n.menut_selected_project)
        self.add_command_item(cmd_name=MT_ADD_IMAGES, info=i18n.menut_add_images)
        self.add_separator()
        self.add_command_item(cmd_name=MT_SELECT_P, info=i18n.menut_select_project)
        self.add_command_item(cmd_name=MT_CLOSE_P, info=i18n.menut_close_project)
        self.add_command_item(cmd_name=MT_DELETE_P, info=i18n.menut_delete_project)
        # separator
        self.add_separator()
        # menu_edit items
        self.add_command_item(cmd_name=MT_SEL_F, info=i18n.menut_selected_file)
        self.add_command_item(cmd_name=MT_OPEN_F, info=i18n.menut_open_file)
        self.add_command_item(cmd_name=MT_RENAME_F, info=i18n.menut_rename_file)
        self.add_command_item(cmd_name=MT_DELETE_F, info=i18n.menut_delete_file)

    def set_selected(self, data):
        logger.debug(f"set selected data {data}")
        self.data = data
        sel_p = i18n.menut_selected_project.copy()
        sel_p['label'] += " " + data["project"]
        sel_f = i18n.menut_selected_file.copy()
        sel_f['label'] += " " + data["file"]
        self.update_command_or_cascade(name=MT_SEL_P, info_updated=sel_p)
        self.update_command_or_cascade(name=MT_SEL_F, info_updated=sel_f)

    def update_language(self):
        logger.debug(f"{self.winfo_name()} update language")
        self.update_command_or_cascade(name=MT_SEL_P, info_updated=i18n.menut_selected_project)
        self.update_command_or_cascade(name=MT_ADD_IMAGES, info_updated=i18n.menut_add_images)
        self.update_command_or_cascade(name=MT_SELECT_P, info_updated=i18n.menut_select_project)
        self.update_command_or_cascade(name=MT_CLOSE_P, info_updated=i18n.menut_close_project)
        self.update_command_or_cascade(name=MT_DELETE_P, info_updated=i18n.menut_delete_project)
        self.update_command_or_cascade(name=MT_SEL_F, info_updated=i18n.menut_selected_file)
        self.update_command_or_cascade(name=MT_OPEN_F, info_updated=i18n.menut_open_file)
        self.update_command_or_cascade(name=MT_DELETE_F, info_updated=i18n.menut_delete_file)
        self.update_command_or_cascade(name=MT_RENAME_F, info_updated=i18n.menut_rename_file)

    def add_command_item(self, cmd_name, info=None):
        if info:
            logger.debug(f"add command item {cmd_name} with info {info}")
            self.__menu_names[cmd_name][M_MASTER] \
                .add_command(label=info.get(M_LABEL, ""), underline=info.get(M_UNDERLINE, -1),
                             state=info.get(M_DEFAULT_STATE, M_STATE_NORMAL),
                             accelerator=info.get(M_ACCELERATOR, ""), command=...)
        else:
            self.__menu_names[cmd_name][M_MASTER] \
                .add_command(label=cmd_name, state=M_STATE_NORMAL, command=...)

    def update_command_or_cascade(self, name, info_updated, update_state=False):
        logger.debug(f"update item {name} with info {info_updated}")
        master = self.__menu_names[name][M_MASTER]
        entry_to_update = self.__menu_names[name][M_INDEX]
        if M_LABEL in info_updated:
            master.entryconfigure(entry_to_update, label=info_updated[M_LABEL])
        if M_UNDERLINE in info_updated:
            master.entryconfigure(entry_to_update, underline=info_updated[M_UNDERLINE])
        if M_STATE in info_updated and update_state:
            master.entryconfigure(entry_to_update, state=info_updated[M_STATE])
        if M_ACCELERATOR in info_updated:
            master.entryconfigure(entry_to_update, accelerator=info_updated[M_ACCELERATOR])
        if M_COMMAND in info_updated:
            master.entryconfigure(entry_to_update, command=info_updated[M_COMMAND])


class ProjectTreeView(ttk.Treeview, View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def create_view(self):
        logger.debug("create view in project tree view")

        self[T_COLUMNS] = (T_SIZE, T_MODIFIED)
        self.column(T_NAME, anchor=T_CENTER, minwidth=150, width=150)
        self.column(T_SIZE, anchor=T_CENTER, minwidth=150, width=150)
        self.column(T_MODIFIED, anchor=T_CENTER, minwidth=150, width=150)
        self.heading(T_NAME, text=i18n.tree_view[T_COLUMNS][T_NAME_HEADING])
        self.heading(T_SIZE, text=i18n.tree_view[T_COLUMNS][T_SIZE])
        self.heading(T_MODIFIED, text=i18n.tree_view[T_COLUMNS][T_MODIFIED])
        # select mode
        self["selectmode"] = "browse"
        # displayColumns
        self["displaycolumns"] = [T_SIZE, T_MODIFIED]
        # show
        self["show"] = "tree headings"
        # tree Display tree labels in column #0.

    def update_language(self):
        logger.debug("update language in project tree view")
        self.heading(T_NAME, text=i18n.tree_view[T_COLUMNS][T_NAME_HEADING])
        self.heading(T_SIZE, text=i18n.tree_view[T_COLUMNS][T_SIZE])
        self.heading(T_MODIFIED, text=i18n.tree_view[T_COLUMNS][T_MODIFIED])


class Viewer3DView(View):

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._title = None
        self.data = None
        self.master = master
        self.__canvas_image: Optional[CanvasImage] = None
        self.__canvas_3d: Optional[Viewer3D] = None
        self.__placeholder = ttk.Frame(self)

    def update_language(self):
        logger.debug("update language in Viewer3D view")

    def create_view(self):
        logger.debug("create_view in Viewer3D view")
        self.rowconfigure(0, weight=1)  # make grid cell expandable
        self.columnconfigure(0, weight=1)
        self.__placeholder.grid(row=0, column=0, sticky='nswe')
        self.__placeholder.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__placeholder.columnconfigure(0, weight=1)

    def __update_view(self, type_of_file):
        logger.debug("update view in selected Viewer3D view")
        for widget in self.__placeholder.winfo_children():
            widget.destroy()
        path = Path(self.data[P_PATH])
        if type_of_file == "image":
            logger.debug("show image in Viewer3D view")
            self.__canvas_image = CanvasImage(placeholder=self.__placeholder, path=path, can_grab_focus=self.master.can_grab_focus())
            self.__canvas_image.grid(row=0, column=0, sticky='nswe')
        elif type_of_file == "obj":
            logger.debug("show mesh in Viewer3D view")
            # self.__canvas_3d = Viewer3D(placeholder=self.__placeholder, obj_file_path=path)
            # self.__canvas_3d.grid(row=0, column=0, sticky='nswe')
            # self.__canvas_3d.update_display()
        else:
            logger.debug("no file")
            # no file

    def update_selected_file(self, data: Optional[dict] = None):
        logger.debug("update selected file in selected file view")
        if data and data != self.data:
            path = Path(data[P_PATH])
            # is image
            if path.suffix in (".png", ".bmp", ".jpg"):
                self.data = data
                self.__update_view(type_of_file="image")
            # is obj
            if path.suffix == ".obj":
                self.data = data
                self.__update_view(type_of_file="obj")


class SelectedFileView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.data = None
        self._file_label = tk.StringVar()
        self._suffix_label = tk.StringVar()
        self._file_label_info = tk.StringVar()
        self._suffix_label_info = tk.StringVar()
        self._no_file = tk.StringVar()

        self._open_s_label = tk.StringVar()
        self._open_s: Optional[ttk.Button] = None

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
        self._open_s = ttk.Button(self, textvariable=self._open_s_label, command=...)

    def __update_view(self):
        logger.debug("update view in selected file view")
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Label):
                widget.destroy()
            else:
                widget.grid_forget()
        if self.data:
            ttk.Label(self, textvariable=self._file_label_info).grid(column=0, row=0, sticky=tk.EW)
            ttk.Label(self, textvariable=self._file_label).grid(column=1, row=0, sticky=tk.EW)
            ttk.Label(self, textvariable=self._suffix_label_info).grid(column=0, row=1, sticky=tk.EW)
            ttk.Label(self, textvariable=self._suffix_label).grid(column=1, row=1, sticky=tk.EW)
            # update buttons
            self._open_s.grid(column=0, row=2, sticky=tk.EW)
        else:
            ttk.Label(self, textvariable=self._no_file).grid(column=0, row=0, sticky=tk.NSEW)

    def set_command(self, btn_name, command):
        logger.debug(f"ModelFit action view set command {command} for {btn_name}")
        if btn_name in self._buttons.keys():
            button: ttk.Button = self._buttons.get(btn_name)()
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


class ProjectInfoView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._project_name = None
        self._name_label_info = tk.StringVar()
        self._path_label_info = tk.StringVar()
        self._name_label = tk.StringVar()
        self._path_label = tk.StringVar()
        self._no_project = tk.StringVar()
        self.update_language()

    def update_language(self):
        logger.debug("update language in project info view")
        self._path_label_info.set(i18n.selected_project_view[PV_PATH])
        self._name_label_info.set(i18n.selected_project_view[PV_NAME])
        self._no_project.set(i18n.selected_project_view[PV_NO])

    def create_view(self):
        logger.debug("create view in project info view")
        for widget in self.winfo_children():
            widget.destroy()
        if self._project_info:
            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            ttk.Label(self, textvariable=self._name_label_info).grid(column=0, row=0, sticky=tk.W)
            ttk.Label(self, textvariable=self._name_label).grid(column=1, row=0, sticky=tk.EW)
            ttk.Label(self, textvariable=self._path_label_info).grid(column=0, row=1, sticky=tk.W)
            ttk.Label(self, textvariable=self._path_label).grid(column=1, row=1, sticky=tk.EW)
        else:
            ttk.Label(self, textvariable=self._no_project).grid(column=0, row=0, sticky=tk.NSEW)

    def __step_done_set_label(self, label, info_to_check):
        logger.debug(f"set {label} of {info_to_check} as done or not")
        if self._project_info.getboolean(info_to_check, P_DONE, fallback=False):
            label.set(i18n.selected_project_view[PV_DONE])
        else:
            label.set(i18n.selected_project_view[PV_NOT_DONE])

    def update_selected_project(self, data: ConfigParser):
        logger.debug("update selected project in project info view")
        self._project_info = data
        if self._project_info:
            self._project_name = str(self._project_info.sections()[0])
            self._path_label.set(str(self._project_info[self._project_name][P_PATH]))
            self._name_label.set(self._project_info[self._project_name][P_NAME])
        self.create_view()


class ModelFitView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.scrollFrame = ScrollFrame(self)  # add a new scrollable frame.
        self.scrollFrame.viewPort.rowconfigure(0, weight=1)
        self.scrollFrame.viewPort.columnconfigure(0, weight=1)
        self._project_info: Optional[ConfigParser] = None
        models = [x for x in Path(nect_config[CONFIG][MODEL_FOLDER]).iterdir() if x.is_dir()]
        self.model_combobox = ComboLabel(master=self.scrollFrame.viewPort, label_text="m_combo",
                                         selected="EMOCA_v2_lr_mse_20",
                                         values=[str(file_name.stem) for file_name in models], state="readonly")
        self.save_images = CheckLabel(master=self.scrollFrame.viewPort, label_text="c_simages", default=True)
        self.save_codes = CheckLabel(master=self.scrollFrame.viewPort, label_text="c_scodes", default=False)
        self.save_mesh = CheckLabel(master=self.scrollFrame.viewPort, label_text="c_smesh", default=False)
        self.fit_mode = RadioList(master=self.scrollFrame.viewPort, list_title="mode_radio", default=R_DETAIL,
                                  data=[R_DETAIL, R_COARSE])

        self.select_images_button_label = tk.StringVar()
        self.select_images_button: Optional[ttk.Button] = None
        self.update_language()

    def update_language(self):
        logger.debug("update language in ModelFit view")
        self.model_combobox.update_language()
        self.save_images.update_language()
        self.save_codes.update_language()
        self.save_mesh.update_language()
        self.fit_mode.update_language()
        self.select_images_button_label.set(i18n.project_actions_fit[MF_SELECT_IMG])

    def create_view(self):
        logger.debug("create view in ModelFit view")
        self.scrollFrame.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=5)
        # Combo box for model selection
        self.model_combobox.create_view()
        # Checkboxes for options
        self.save_images.create_view()
        self.save_codes.create_view()
        self.save_mesh.create_view()
        self.fit_mode.create_view()
        # Button to select images
        self.select_images_button = ttk.Button(self.scrollFrame.viewPort, textvariable=self.select_images_button_label)

    def __update_view(self):
        logger.debug("update view in ModelFit view")
        self.model_combobox.grid(row=0, column=0, sticky=tk.EW, pady=5)
        self.save_images.grid(row=1, column=0, sticky=tk.EW, pady=5)
        self.save_codes.grid(row=2, column=0, sticky=tk.EW, pady=5)
        self.save_mesh.grid(row=3, column=0, sticky=tk.EW, pady=5)
        self.fit_mode.grid(row=4, column=0, sticky=tk.EW, pady=5)
        self.select_images_button.grid(row=5, column=0, sticky=tk.E, pady=5)

    def update_selected_project(self, data: Optional[ConfigParser] = None):
        logger.debug("update selected project in scan view")
        self._project_info = data
        self.__update_view()

    def get_form(self):
        return {
            MF_MODEL: self.model_combobox.get_value(),
            MF_SAVE_IMAGES: self.save_images.get_value(),
            MF_SAVE_CODES: self.save_codes.get_value(),
            MF_SAVE_MESH: self.save_mesh.get_value(),
            MF_FIT_MODE: self.fit_mode.get_value()
        }


class AURecognitionView(View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info: Optional[ConfigParser] = None
        self._exist_label_info = tk.StringVar()
        self._exist_label: Optional[ttk.Label] = None
        self.update_language()

    def create_view(self):
        logger.debug("update view in AURecognition view")

    def update_language(self):
        logger.debug("update language in AURecognition view")

    def update_selected_project(self, data=None):
        logger.debug("update selected project in AURecognition view")
        self._project_info = data


class ProjectActionView(ttk.Notebook, View):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._project_info = None
        self._model_fit_frame = ModelFitView(self)
        self._au_recognition_frame = AURecognitionView(self)

    def update_language(self):
        logger.debug(f"update language in project action view")
        self.tab(self._model_fit_frame, text=i18n.project_actions_fit[PA_NAME])
        self.tab(self._au_recognition_frame, text=i18n.project_actions_au_rec[PA_NAME])
        self._model_fit_frame.update_language()
        self._au_recognition_frame.update_language()

    def __tabs_change_state(self, new_state):
        logger.debug(f"set all action tabs in state: {new_state}")
        for i, item in enumerate(self.tabs()):
            self.tab(item, state=new_state)

    def create_view(self):
        logger.debug(f"create view in project action view")
        self.add(self._model_fit_frame, text=i18n.project_actions_fit[PA_NAME])
        self.add(self._au_recognition_frame, text=i18n.project_actions_au_rec[PA_NAME])
        self.__update_view()

    def __update_view(self):
        logger.debug(f"update view in project action view")
        if not self._project_info:
            self.__tabs_change_state(PA_DISABLED)
        else:
            self.__tabs_change_state(PA_NORMAL)
            self.select(self._model_fit_frame)

    def bind_controllers(self, model_fit_controller: Controller, au_recognition_controller: Controller):
        logger.debug(f"bind controllers in project action view")
        self.__bind_controller(controller=model_fit_controller, view=self._model_fit_frame)
        self.__bind_controller(controller=au_recognition_controller, view=self._au_recognition_frame)

    @staticmethod
    def __bind_controller(controller: Controller, view: View):
        logger.debug(f"project action bind {controller.__class__} to {view.__class__}")
        controller.bind(view)

    def update_selected_project(self, data=None):
        logger.debug(f"update selected project in project action view")
        self._project_info = data
        self.__update_view()
