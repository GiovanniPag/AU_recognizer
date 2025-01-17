import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class FloatPicker(View):
    def __init__(self, master=None, label_text="no_text", default=0.0, min_value=0.01, max_value=1, increment=0.01,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._label_info = tk.StringVar(value=label_text)
        self.float_value = tk.DoubleVar(value=default)
        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment
        self.update_language()

    def create_view(self):
        ttk.Label(self, textvariable=self._label_info).pack(side=tk.LEFT)
        spinbox_frame = tk.Frame(self, padx=5)
        spinbox_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        spinbox = ttk.Spinbox(spinbox_frame, from_=self.min_value, to=self.max_value, increment=self.increment,
                              textvariable=self.float_value, width=5)
        spinbox.pack(side=tk.LEFT)

    def get_value(self):
        return self.float_value.get()

    def update_language(self):
        # Update the label text according to the current language or any other mechanism
        self._label_info.set(self.label_text)
