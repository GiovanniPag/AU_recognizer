import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class IntPicker(View):
    def __init__(self, master=None, label_text="no_text", default=0, min_value=0, max_value=100, increment=1, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.int_value = tk.IntVar(value=default)
        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment
        self.update_language()

    def create_view(self):
        ttk.Label(self, textvariable=self._label_info).pack(side=tk.LEFT)
        spinbox_frame = tk.Frame(self, padx=5)
        spinbox_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        spinbox = tk.Spinbox(spinbox_frame, from_=self.min_value, to=self.max_value, increment=self.increment,
                             textvariable=self.int_value, width=5)
        spinbox.pack(side=tk.LEFT)

    def get_value(self):
        return self.int_value.get()

    def update_language(self):
        self._label_info.set(i18n.entry_buttons[self.label_text])
