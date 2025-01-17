import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class CheckLabel(View):
    def __init__(self, master=None, label_text="no_text", default=False, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.value = tk.BooleanVar(value=default)
        self.command = command
        self.update_language()

    def create_view(self):
        check_box = ttk.Checkbutton(self, textvariable=self._path_label_info, variable=self.value, command=self.command)
        check_box.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def get_value(self):
        return self.value.get()

    def update_language(self):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])
