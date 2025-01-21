from tkinter import ttk as ttk

from AU_recognizer.core.user_interface import ThemeManager, AppearanceModeTracker
from AU_recognizer.core.user_interface.views.view import View
from AU_recognizer.core.util import logger, T_COLUMNS, T_SIZE, T_MODIFIED, T_NAME, T_CENTER, i18n, T_NAME_HEADING


class ProjectTreeView(ttk.Treeview, View):
    def __init__(self, master=None):
        View.__init__(self, master=master)
        ttk.Treeview.__init__(self, master, columns=(T_NAME, T_SIZE, T_MODIFIED))
        self.tree_style = None
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.update_style()
        AppearanceModeTracker.add(self.update_style)

    def create_view(self):
        logger.debug("create view in project tree view")
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

    def update_style(self, _="dark"):
        bg_color = self._apply_appearance_mode(ThemeManager.theme["CustomFrame"]["fg_color"])
        text_color = self._apply_appearance_mode(ThemeManager.theme["CustomLabel"]["text_color"])
        selected_color = self._apply_appearance_mode(ThemeManager.theme["CustomButton"]["fg_color"])
        self.tree_style = ttk.Style()
        self.tree_style.theme_use('default')
        self.tree_style.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color,
                                  borderwidth=0)
        self.tree_style.map('Treeview', background=[('selected', bg_color)], foreground=[('selected', selected_color)])
