import tkinter as tk
from typing import Callable, Union

from AU_recognizer.core.user_interface import CustomLabel, CustomFrame, CustomButton
from AU_recognizer.core.user_interface.dialogs.color_picker_dial import AskColor
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import logger, i18n


class ColorPickerLabel(View):
    def __init__(self, master=None, label_text="no_text", default="#FFFFFF", on_change: Union[Callable, None] = None,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self.on_change = on_change
        self._label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.color = tk.StringVar(value=default)
        self.color_button = None
        self.update_language()

    def create_view(self):
        CustomLabel(self, textvariable=self._label_info).pack(side=tk.LEFT)
        button_frame = CustomFrame(self)
        button_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.color_button = CustomButton(button_frame, text="", command=self.pick_color, width=28,
                                         height=28)
        self.color_button.pack(side=tk.LEFT)
        self.color_button.configure(fg_color=self.color.get())

    def pick_color(self):
        logger.debug("pick color")
        color = AskColor(master=self.master, command=lambda value: (self.color.set(value), self.on_change()),
                         initial_color=self.color.get()).get()
        if color and self.color_button.winfo_exists():
            logger.debug(f"color picked {color}")
            self.color.set(color)
            self.color_button.configure(fg_color=self.color.get())

    def get_value(self):
        return self.color.get()

    def update_language(self):
        self._label_info.set(i18n.entry_buttons[self.label_text])
