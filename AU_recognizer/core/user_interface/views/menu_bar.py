from tkinter import StringVar, Menu

from AU_recognizer.core.user_interface import AppearanceModeTracker, ThemeManager
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import M_HELP, M_MASTER, M_FILE, M_INDEX, M_EDIT, M_NEW, M_OPEN, M_SETTINGS, \
    M_EXIT, M_LANGUAGE, M_ABOUT, M_LOGS, M_GUIDE, M_IT, M_RADIO, M_VARIABLE, M_VALUE, M_EN, logger, \
    i18n, M_LABEL, M_UNDERLINE, M_DEFAULT_STATE, M_STATE_NORMAL, M_ACCELERATOR, M_STATE, M_COMMAND


class MenuBar(Menu, View):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.menu_file = Menu(self)
        self.menu_help = Menu(self, name=M_HELP)
        self.menu_help_language = Menu(self.menu_help)
        self.update_style()
        AppearanceModeTracker.add(self.update_style)
        self.language = StringVar()

        self.__menu_names = {
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

    def update_style(self, _="dark"):
        bg_color = self._apply_appearance_mode(ThemeManager.theme["DropdownMenu"]["fg_color"])
        fg_color = self._apply_appearance_mode(ThemeManager.theme["DropdownMenu"]["text_color"])
        hover_color = self._apply_appearance_mode(ThemeManager.theme["DropdownMenu"]["hover_color"])
        self.configure(background=bg_color)
        self.configure(foreground=fg_color)
        self.configure(borderwidth=0)
        self.configure(activebackground=hover_color)
