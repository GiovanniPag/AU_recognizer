import ast
import re
import shutil
from collections import namedtuple
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog
from typing import Optional

from PIL import Image

from AU_recognizer import open_message_dialog, open_confirmation_dialogue, delete_path, rename_path, \
    confirm_rename_path
from AU_recognizer.core.controllers import Controller
from AU_recognizer.core.projects import store_open_project, add_to_open_projects, create_project_folder, rename_project
from AU_recognizer.core.util import open_guide, open_log_folder, check_if_folder_exist, check_if_is_project, \
    open_path_by_os, check_if_file_exist, get_desktop_path
from AU_recognizer.core.util.config import logger, nect_config, purge_option_config
from AU_recognizer.core.util.constants import *
from AU_recognizer.core.util.language_resource import i18n
from AU_recognizer.core.views import sizeof_fmt
from AU_recognizer.core.views.dialog import DialogProjectOptions, SettingsDialog, SelectFitImageDialog
from AU_recognizer.core.views.view import MenuBar, ProjectTreeView, Viewer3DView, SelectedFileView, ProjectInfoView, \
    ProjectActionView, ModelFitView, AURecognitionView, TreeViewMenu


class MenuController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.master = master
        self.view = None

    def bind(self, v: MenuBar):
        logger.debug("bind in menu controller")
        self.view = v
        self.view.create_view()
        self.view.language.set(nect_config[CONFIG][LANGUAGE])
        # bind command to menu items
        logger.debug("menu controller bind commands")
        # menu help
        self.view.update_command_or_cascade(M_ABOUT, {M_COMMAND: lambda: open_message_dialog(self.master, "about")})
        self.view.update_command_or_cascade(M_LOGS, {M_COMMAND: lambda: open_log_folder()})
        self.view.update_command_or_cascade(M_GUIDE, {M_COMMAND: lambda: open_guide()})
        # menu help language
        self.view.update_command_or_cascade(M_IT, {M_COMMAND: lambda: self.language_change(M_IT)})
        self.view.update_command_or_cascade(M_EN, {M_COMMAND: lambda: self.language_change(M_EN)})
        # menu file
        self.view.update_command_or_cascade(M_NEW, {M_COMMAND: lambda: self.create_new_project()})
        self.view.update_command_or_cascade(M_OPEN, {M_COMMAND: lambda: self.open_project()})
        self.view.update_command_or_cascade(M_SETTINGS, {M_COMMAND: lambda: self.open_settings_dialog()})
        self.view.update_command_or_cascade(M_EXIT, {M_COMMAND: lambda: self.close_app()})

    def open_project(self):
        logger.debug("open project")
        opening = True
        while opening:
            path = self.choose_folder()
            if path != "":
                logger.debug("chosen project path: " + str(path))
                is_project, metadata = check_if_is_project(path, path.name)
                if is_project:
                    if str(path) in self.master.open_projects:
                        logger.debug("project is already open")
                        open_message_dialog(self.master, "project_already_open", ERROR_ICON)
                    else:
                        logger.debug("project opened successfully")
                        self.master.add_project(metadata, path.name)
                        add_to_open_projects(path)
                        open_message_dialog(self.master, "project_open_success")
                        logger.debug("menu_bar <<UpdateTree>> event generation")
                        self.master.event_generate("<<UpdateTree>>")
                        opening = False
                else:
                    logger.debug("selected folder is not a project")
                    open_message_dialog(self.master, "not_project", ERROR_ICON)
            else:
                opening = False

    def create_new_project(self):
        logger.debug("create new project")
        creating = True
        while creating:
            path = self.choose_folder()
            if path != "":
                logger.debug("chosen project path: " + str(path))
                name = self.ask_project_name()
                if name != i18n.dialog_buttons[I18N_BACK_BUTTON]:
                    if name != "":
                        logger.debug("chosen project name: " + name)
                        exist = check_if_folder_exist(path, name)
                        if not exist:
                            create_project_folder(path / name)
                            metadata = store_open_project(path, name)
                            self.master.add_project(metadata, name)
                            logger.debug("menu_bar <<UpdateTree>> event generation")
                            self.master.event_generate("<<UpdateTree>>")
                            logger.debug("project created successfully")
                        else:
                            logger.debug("invalid name")
                            open_message_dialog(self.master, "project_exist", ERROR_ICON)
                    creating = False
                else:
                    logger.debug("go back to choose path")
            else:
                creating = False

    def close_app(self):
        logger.debug("close app")
        self.master.destroy()

    def ask_project_name(self):
        logger.debug("open p_options dialog, ask project name")
        p_name = DialogProjectOptions(master=self.master, has_back=True).show()
        return p_name

    def choose_folder(self):
        logger.debug("choose directory")
        dir_name = filedialog.askdirectory(initialdir=nect_config[CONFIG][PROJECTS_FOLDER], mustexist=True,
                                           parent=self.master, title=i18n.choose_folder_dialog[I18N_TITLE])
        if dir_name != () and dir_name:
            dir_name = Path(dir_name)
            logger.debug("chosen path: " + str(dir_name))
            return dir_name
        else:
            logger.debug("abort choosing")
            return ""

    def language_change(self, language):
        changed = i18n.change_language(language)
        if changed:
            logger.debug("menu_bar <<LanguageChange>> event generation")
            self.master.event_generate("<<LanguageChange>>")

    def open_settings_dialog(self):
        logger.debug("open settings dialog")
        SettingsDialog(master=self.master).show()


class TreeViewMenuController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.master = master
        self.controller: Optional[TreeController] = None
        self.view = None
        self.after_id = None

    def bind(self, v: TreeViewMenu):
        logger.debug("bind in menu controller")
        self.view = v
        self.view.create_view()
        # bind command to menu items
        logger.debug("menu controller bind commands")
        self.view.bind("<<Unpost>>", self.unpost_menu)
        # menu help
        self.view.update_command_or_cascade(MT_ADD_IMAGES, {M_COMMAND: lambda: self.ask_images()})
        self.view.update_command_or_cascade(MT_SELECT_P, {M_COMMAND: lambda: self.controller.select_project(None)})
        self.view.update_command_or_cascade(MT_CLOSE_P, {M_COMMAND: lambda: self.close_project()})
        self.view.update_command_or_cascade(MT_DELETE_P, {M_COMMAND: lambda: self.delete_project()})
        self.view.update_command_or_cascade(MT_OPEN_F, {M_COMMAND: lambda: self.controller.open_file(None)})
        self.view.update_command_or_cascade(MT_DELETE_F, {M_COMMAND: lambda: self.delete_file()})
        self.view.update_command_or_cascade(MT_RENAME_F, {M_COMMAND: lambda: self.rename_file()})

    def ask_images(self):
        logger.debug(f"add images to  project {self.view.data[TM_PROJECT]}")
        # select files from file system, Open a file dialog to choose image files
        file_paths = filedialog.askopenfilenames(
            initialdir=get_desktop_path(),
            parent=self.master,
            title=i18n.choose_images_dialog[I18N_TITLE],
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")),
            multiple=True
        )
        self.add_images(file_paths)

    def add_images(self, file_paths):
        if file_paths:
            logger.debug(f"selected files {file_paths}")
            destination_dir = Path(self.view.data[TM_PROJECT_PATH]) / F_INPUT
            # Check if destination directory exists
            if destination_dir.is_dir():
                # import img files to INPUT folder copy as png
                for file_path in file_paths:
                    source_path = Path(file_path)
                    filename = source_path.name
                    logger.debug(f"try to import: {source_path} inside {destination_dir}")
                    # Convert images to png if needed else copy
                    if source_path.suffix.lower() in ['.jpg', '.png', '.bmp']:
                        shutil.copy(source_path, destination_dir)
                        logger.info(f"Success: {filename} copied successfully inside {destination_dir}")
                    elif source_path.suffix.lower() in ['.jpeg', '.gif']:
                        try:
                            filestem = source_path.stem
                            with Image.open(source_path) as img:
                                img.save(destination_dir / (filestem + ".png"), 'PNG', optimize=True, quality=100)
                                logger.info(
                                    f"Success: {filename} copied and converted to png successfully inside {destination_dir}")
                        except Exception as e:
                            logger.error(f"Error converting {filename} to PNG: {e}")
                    else:
                        logger.error(f"{source_path} can't be imported into program")
            else:
                logger.error("Input folder of project does not exist")

    def delete_project(self, confirmation=True):
        logger.debug(
            f"close project {self.view.data[TM_PROJECT]} and remove the folder and all it's content from file system")
        answer = None
        if confirmation:
            answer = open_confirmation_dialogue(self.master, "delete_project")
        if not confirmation or answer == I18N_YES_BUTTON:
            self.close_project()
            delete_path(path_to_delete=self.view.data[TM_PROJECT_PATH], ask_confirmation=False, master=self.master)

    def delete_file(self):
        logger.debug(f"remove file {self.view.data[TM_FILE]} from file system")
        if self.view.data[TM_FILE_PATH] == self.view.data[TM_PROJECT_PATH]:
            self.delete_project()
        else:
            if self.view.data[TM_FILE] == (self.view.data[TM_PROJECT] + ".ini"):
                answer = open_confirmation_dialogue(self.master, "corrupt_project")
                if answer == I18N_YES_BUTTON:
                    self.close_project()
                    delete_path(path_to_delete=self.view.data[TM_FILE], ask_confirmation=False,
                                master=self.master)
            else:
                removed = delete_path(path_to_delete=self.view.data[TM_FILE_PATH], master=self.master)
                if removed:
                    is_project = check_if_is_project(path=Path(self.view.data[TM_PROJECT_PATH]),
                                                     project_name=self.view.data[TM_PROJECT])
                    if is_project:
                        self.controller.remove_tree_node(path=self.view.data[TM_FILE_PATH])
                    else:
                        open_message_dialog(master=self.master, message="corrupted_project", icon=ERROR_ICON)
                        self.close_project()

    def rename_project(self):
        logger.debug(f"rename project {self.view.data[TM_PROJECT_PATH]}")
        path = Path(self.view.data[TM_PROJECT_PATH])
        if path.exists():
            new_project_name = DialogProjectOptions(master=self.master).show()
            if new_project_name != "":
                logger.debug("chosen project name: " + new_project_name)
                exist = check_if_folder_exist(path.parent, new_project_name) or check_if_file_exist(path=path,
                                                                                                    file_name=(
                                                                                                            new_project_name + ".ini"))
                if not exist:
                    # project ini,
                    rename_path(path_to_rename=(path / (self.view.data[TM_PROJECT] + ".ini")),
                                new_name=new_project_name)
                    # rename project folder,
                    new_path = rename_path(path_to_rename=path, new_name=new_project_name)
                    # modify metadata
                    rename_project(config_path=(new_path / (new_project_name + ".ini")),
                                   old_name=self.view.data[TM_PROJECT], new_name=new_project_name)
                    # modify master list
                    is_project, metadata = check_if_is_project(new_path, new_project_name)
                    self.master.rename_project(old_path=str(path), new_name=new_project_name, new_metadata=metadata)
                    self.master.event_generate("<<UpdateTree>>")
                else:
                    logger.debug("invalid name")
                    open_message_dialog(self.master, "name_taken", ERROR_ICON)
        else:
            logger.error(f"Error: The specified path '{path}' does not exist.")

    def rename_file(self):
        logger.debug(f"rename file {self.view.data[TM_FILE]}")
        if self.view.data[TM_FILE_PATH] == self.view.data[TM_PROJECT_PATH] or self.view.data[TM_FILE] == (
                self.view.data[TM_PROJECT] + ".ini"):
            self.rename_project()
        else:
            renamed, path = confirm_rename_path(path_to_rename=self.view.data[TM_FILE_PATH], master=self.master)
            if renamed:
                self.controller.rename_tree_node(renamed_path=self.view.data[TM_FILE_PATH], new_path=path)

    def close_project(self):
        logger.debug(f"close project {self.view.data}")
        selp = self.controller.get_last_selected_project()
        if selp is not None and selp[P_PATH] == self.view.data[TM_PROJECT_PATH]:
            self.controller.deselect_all()
        else:
            self.controller.deselect_file()
        self.controller.remove_tree_node(path=self.view.data[TM_PROJECT_PATH])
        self.master.remove_project(self.view.data[TM_PROJECT_PATH])

    def open_menu(self, event, data):
        logger.debug(f"open tree view menu at {event.__dict__} with data {data}")
        self.view.set_selected(data)
        self.view.post(event.x_root, event.y_root)
        self.view.focus_set()
        self.view.bind_all("<Button-1>", self.on_close)
        self.view.bind_all("<Button-2>", self.on_close)
        self.view.bind("<FocusOut>", self.on_close)
        self.after_id = self.view.after(100, lambda: self.view.bind_all("<Button-3>", self.on_close))

    def on_close(self, _):
        logger.debug(f"on_close of tree view menu {_}")
        unpost = True
        if _.type == "4":
            # Check if the pointer is over the menubar
            menubar_bbox = re.split("[x+]", self.view.winfo_geometry())  # widthxheight+x+y
            menubar_width, menubar_height, menubar_x, menubar_y = map(int, menubar_bbox)
            if menubar_x <= _.x_root <= menubar_x + menubar_width and menubar_y <= _.y_root <= menubar_y + menubar_height:
                unpost = False
        if unpost:
            self.view.event_generate("<<Unpost>>", when="head")

    def unpost_menu(self, _):
        logger.debug(f"unpost_menu of tree view menu {_}")
        self.view.unbind_all("<Button-1>")
        self.view.unbind_all("<Button-2>")
        self.view.after_cancel(self.after_id)
        self.view.unbind_all("<Button-3>")
        self.view.unpost()


class Viewer3DController(Controller):

    def __init__(self, master=None) -> None:
        super().__init__()
        self.master = master
        self.view = None
        self.data = None

    def bind(self, v: Viewer3DView):
        logger.debug("bind in Viewer3D controller")
        self.view = v
        self.view.create_view()

    def update_view(self, data: Optional[dict] = None):
        logger.debug(f"update view in Selected file controller")
        self.view.update_selected_file(data)
        if data:
            self.data = data
        else:
            self.data = None


class TreeController(Controller):

    def __init__(self, context_controller: TreeViewMenuController, master=None) -> None:
        super().__init__()
        self.master = master
        self.context_controller = context_controller
        self.view: ProjectTreeView or None = None
        self.__last_selected_project = None
        self.__last_selected_file = None
        self.menu_pos = None

    def get_last_selected_project(self):
        logger.debug(f"return last selected project path")
        return self.__last_selected_project or None

    def get_last_selected_file(self):
        logger.debug(f"return last selected file path")
        return self.__last_selected_file or None

    def bind(self, v: ProjectTreeView):
        logger.debug("bind in tree view")
        self.view = v
        self.view.create_view()
        logger.debug("Tree_view controller bind commands")
        self.view.bind("<<TreeviewSelect>>", self.focus_item)
        self.view.tag_bind('item', '<Double-Button-1>', self.select_project)
        self.view.tag_bind('file', '<Double-Button-1>', self.open_file)
        self.view.bind('<2>', self.deselect)
        self.view.bind('<<contextual_menu>>', self.show_contextual_menu)
        self.view.tag_bind('item', "<Button-3>", self.show_contextual_menu_trigger)
        logger.debug("populate treeview items")
        self.update_tree_view(populate_root=True)

    def show_contextual_menu_trigger(self, _):
        logger.debug(f"show contextual menu of treeview trigger {_}")
        self.menu_pos = _
        self.view.event_generate("<<contextual_menu>>", when="tail")

    def show_contextual_menu(self, _):
        logger.debug("show contextual menu of treeview")
        item_id = self.view.identify("item", self.menu_pos.x, self.menu_pos.y)
        root = self.get_root_node(item_id)
        root_data = ast.literal_eval(root[T_VALUES][-1])
        data = {TM_FILE: Path(item_id).name,
                TM_PROJECT: root[T_TEXT],
                TM_FILE_PATH: item_id,
                TM_PROJECT_PATH: root_data[P_PATH]
                }
        self.view.selection_set(item_id)
        self.context_controller.open_menu(self.menu_pos, data)

    def deselect(self, _):
        logger.debug(f"deselect current item, event {_}")
        # Get the selected items
        selected_items = self.view.selection()
        # Deselect all selected items
        for item in selected_items:
            self.view.selection_remove(item)

    def focus_item(self, _):
        self.schedule_update()
        selection = self.view.selection()
        if selection:
            selected_item_id = selection[0]
            selected_item = self.view.item(selected_item_id)
            logger.debug(f"selected item {selected_item}")
            self.__last_selected_file = ast.literal_eval(selected_item[T_VALUES][-1])
            self.master.event_generate("<<selected_item>>")
        else:
            logger.debug("tree view: no selected file")
            self.__last_selected_file = {}
            self.master.event_generate("<<selected_item>>")

    def open_file(self, _):
        self.master.event_generate("<<open_file>>")

    def deselect_all(self):
        self.schedule_update()
        logger.debug("tree view deselect all")
        self.deselect_file()
        self.deselect_project()

    def deselect_file(self):
        logger.debug("tree view deselect project")
        self.deselect(None)
        self.__last_selected_file = None
        self.master.event_generate("<<selected_file>>")

    def deselect_project(self):
        logger.debug("tree view deselect project")
        self.__last_selected_project = None
        self.master.event_generate("<<selected_project>>")

    def select_project(self, _):
        logger.debug(f"tree view double click: {_}")
        selection = self.view.selection()
        if selection:
            selected_item_id = selection[0]
            selected_item = self.view.item(selected_item_id)
            logger.debug(f"selected project of item: {selected_item}")
            root = self.get_root_node(selected_item_id)
            values = ast.literal_eval(root[T_VALUES][-1])
            self.__last_selected_project = values
            self.master.event_generate("<<selected_project>>")
        else:
            logger.debug("tree view double click: no selected project")
            self.__last_selected_project = None
            self.master.event_generate("<<selected_project>>")

    def get_root_node(self, start_node):
        root_id = start_node
        while "root" not in self.view.item(root_id, "tags"):
            root_id = self.view.parent(root_id)
        logger.debug(f"root node of {start_node} is: {root_id}")
        return self.view.item(root_id)

    def schedule_update(self, _=None):
        logger.debug("Tree_view controller event to update view")
        self.update_tree_view()

    def update_tree_view(self, populate_root=False):
        logger.debug(f"Tree_view controller update view, populate_root = {populate_root}")
        self.nodes_consistency_check()
        if populate_root:
            self.populate_root_nodes()
        self.update_tree_nodes()

    def remove_tree_node(self, path):
        if self.view.exists(path):
            self.view.delete(path)

    def rename_tree_node(self, renamed_path, new_path):
        logger.debug(f"rename {renamed_path} with {new_path}")
        path = Path(new_path)
        if self.view.exists(renamed_path):
            self.view.delete(renamed_path)
            self.add_or_update_item(path=new_path, parent=path.parent)
            if path.is_dir():
                self.__recursive_update_tree(new_path)

    def populate_root_nodes(self):
        logger.debug("Tree_view controller populate root nodes: fetch open projects")
        data = self.master.open_projects
        for project in data:
            logger.debug(
                f"Tree_view controller populate root nodes: for each project add to tree_nodes and view : {project}")
            p_path = Path(project)
            self.add_or_update_item(path=p_path)

    def add_or_update_item(self, path, parent=None):
        logger.debug(f"Tree_view controller add or update node: {path}, with parent: {parent}")
        item_tags = ("item",) + (("file",) if path.is_file() else ("folder",))
        if parent is None or not self.view.exists(parent):
            item_tags += ("root",)
            parent = None
        data_values = {
            T_SIZE: sizeof_fmt(path.stat().st_size) if path.is_file() else "--",
            T_MODIFIED: datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
                '%Y-%m-%d %H:%M'),
            T_NAME: path.name,
            T_DATA: {
                P_PATH: str(path.resolve()),
                T_FILE_TYPE: path.suffix
            }
        }
        if not self.view.exists(path):
            self.view.insert(parent="" if parent is None else parent,
                             id=path,
                             index=T_END,
                             text=data_values[T_NAME],
                             values=list(data_values.values()), tags=item_tags)
        else:
            self.view.item(path, text=data_values[T_NAME],
                           values=list(data_values.values()), tags=item_tags)

    def update_tree_nodes(self):
        logger.debug("Tree_view controller update tree nodes: for each root node recursive update tree")
        for root_node in self.view.get_children(""):
            self.__recursive_update_tree(root_node)

    def __recursive_update_tree(self, node_to_explore):
        item = self.view.item(node_to_explore)
        data = ast.literal_eval(item[T_VALUES][-1])
        for p in Path(data[P_PATH]).glob('*'):
            self.add_or_update_item(path=p, parent=node_to_explore)
            if p.is_dir():
                self.__recursive_update_tree(p)

    def nodes_consistency_check(self):
        logger.debug("Tree_view controller nodes_consistency_check: for each root node recursive check")
        for root_node in self.view.get_children(""):
            self.__recursive_consistency_check(root_node)

    def __recursive_consistency_check(self, node_to_check):
        item = self.view.item(node_to_check)
        data = ast.literal_eval(item[T_VALUES][-1])
        if Path(data[P_PATH]).exists():
            for child_node in self.view.get_children(node_to_check):
                self.__recursive_consistency_check(child_node)
        else:
            self.remove_tree_node(data[P_PATH])


class SelectedFileController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None

    def bind(self, v: SelectedFileView):
        logger.debug(f"bind in Selected file controller")
        self.view = v
        self.view.create_view()

    def open_file(self, path: Optional[Path] = None):
        logger.debug(f"open file in selected file controller")
        if path:
            logger.debug(f"open file {path}")
            open_path_by_os(path)
        else:
            if self.data:
                logger.debug(f"open last selected file {self.data}")
                path = Path(self.data[P_PATH])
                open_path_by_os(path)
            else:
                logger.warning(f"no file given in open_file")

    def update_view(self, data: Optional[dict] = None):
        logger.debug(f"update view in Selected file controller")
        self.view.update_selected_file(data)
        if data:
            self.data = data
            path = Path(data[P_PATH])
            self.view.set_command(FV_OPEN_S, lambda: self.open_file(path))
        else:
            self.data = None
            self.view.set_command(FV_OPEN_S, ...)


class SelectedProjectController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master

    def bind(self, v: ProjectInfoView):
        logger.debug(f"bind in Selected project controller")
        self.view = v
        self.view.create_view()

    def update_view(self, data):
        logger.debug(f"update view in Selected project controller")
        self.view.update_selected_project(data)


class ModelFitController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self.data = None
        self.scanning = False

    def bind(self, v: ModelFitView):
        logger.debug(f"bind in ModelFit controller")
        self.view = v
        self.view.create_view()
        self.view.select_images_button.config(command=lambda: self.open_image_select_dialog(data=self.view.get_form()))

    def update_selected(self, data):
        logger.debug(f"update selected in ModelFit controller")
        if not self.data or (self.data and self.data != data):
            self.data = data
            self.view.update_selected_project(data)

    def open_image_select_dialog(self, data=None):
        logger.debug("open select images for fit dialog")
        if data:
            SelectFitImageDialog(master=self.master, data=data, project=self.data).show()


class AURecognitionController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master

    def bind(self, v: AURecognitionView):
        logger.debug(f"bind in AURecognition controller")
        self.view = v
        self.view.create_view()

    def update_selected(self, data):
        logger.debug(f"update selected in AURecognition controller")
        self.view.update_selected_project(data)


class ProjectActionController(Controller):
    def __init__(self, master=None) -> None:
        super().__init__()
        self.view = None
        self.master = master
        self._model_fit_controller = ModelFitController(master)
        self._au_recognition_controller = AURecognitionController(master)

    def bind(self, v: ProjectActionView):
        logger.debug(f"bind in project action controller")
        self.view = v
        self.view.create_view()
        self.view.bind_controllers(model_fit_controller=self._model_fit_controller,
                                   au_recognition_controller=self._au_recognition_controller)

    def select_project(self, data):
        logger.debug(f"update selected project in project action controller")
        self.view.update_selected_project(data)
        self._model_fit_controller.update_selected(data)
        self._au_recognition_controller.update_selected(data)
