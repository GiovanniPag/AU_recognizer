from tkinter import Menu

from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import (MT_SEL_P, M_MASTER, M_INDEX, MT_ADD_IMAGES, MT_SELECT_P, MT_CLOSE_P, MT_DELETE_P,
                                     MT_SEL_F, MT_OPEN_F, MT_RENAME_F, MT_DELETE_F, logger, i18n, M_LABEL, M_UNDERLINE,
                                     M_DEFAULT_STATE, M_STATE_NORMAL, M_ACCELERATOR, M_STATE, M_COMMAND)


class TreeViewMenu(Menu):

    def __init__(self, master=None):
        Menu.__init__(self, master, tearoff=0)
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
