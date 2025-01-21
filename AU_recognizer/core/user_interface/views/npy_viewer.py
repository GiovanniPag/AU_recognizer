import tkinter as tk

import numpy as np

from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.user_interface.widgets.core_widget_classes import CustomTextbox


class NpyViewer(View):
    def __init__(self, master=None, path=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.path = path

    def create_view(self):
        # Text widget for displaying the data
        text_widget = CustomTextbox(self, wrap=tk.WORD)
        text_widget.grid(column=0, row=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        data = np.load(self.path)
        text_widget.delete("1.0", tk.END)  # Clear any previous content
        text_widget.insert(tk.END, str(data))  # Insert the data as text

    def update_language(self):
        pass
