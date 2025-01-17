import tkinter as tk
from tkinter import ttk as ttk, colorchooser

from AU_recognizer.core.util import logger
from AU_recognizer.core.views import View


class ColorPicker(View):
    def __init__(self, master=None, label_text="no_text", default="#FFFFFF", **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.color = tk.StringVar(value=default)
        self.color_button = None
        self.update_language()

    def create_view(self):
        ttk.Label(self, textvariable=self._label_info).pack(side=tk.LEFT)
        button_frame = tk.Frame(self, padx=5)
        button_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.color_button = tk.Button(button_frame, text="", relief='flat', command=self.pick_color, width=12, height=1)
        self.color_button.pack(side=tk.LEFT)
        self.color_button['bg'] = self.color.get()
        self.color_button['activebackground'] = self.color.get()

    def pick_color(self):
        logger.debug("pick color")
        color = colorchooser.askcolor(initialcolor=self.color.get())
        if color[1]:
            logger.debug(f"color picked {color}")
            self.color.set(color[1])
            self.color_button['bg'] = color[1]
            self.color_button['activebackground'] = color[1]

    def get_value(self):
        return self.color.get()

    def update_language(self):
        self._label_info.set(i18n.entry_buttons[self.label_text])
