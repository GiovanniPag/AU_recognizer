import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class ScaleLabel(View):
    def __init__(self, master=None, label_text="no_text", from_=1.1, to=1000.0, initial_value=1.1, increment=0.1,
                 orient=tk.HORIZONTAL,
                 command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self._scale_slider = None
        self.from_ = from_
        self.to = to
        self.increment = increment
        self.value = tk.DoubleVar(value=initial_value)
        self.orient = orient
        self.command = command
        self.update_language()

    def create_view(self):
        ttk.Label(self, textvariable=self._path_label_info).pack(side=tk.LEFT)
        scale_frame = tk.Frame(self)
        scale_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._scale_slider = ttk.Scale(scale_frame, from_=self.from_, to=self.to, orient=self.orient,
                                       command=self.command, variable=self.value)
        self._scale_slider.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def get(self):
        return self._scale_slider.get()

    def set(self, new_value):
        self._scale_slider.set(new_value)

    def update_language(self):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])
