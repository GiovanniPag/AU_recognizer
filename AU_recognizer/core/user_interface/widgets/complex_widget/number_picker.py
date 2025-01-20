import tkinter as tk
from typing import Union

from AU_recognizer.core.user_interface import CustomLabel
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.user_interface.widgets.core_widget_classes import CustomSpinbox
from AU_recognizer.core.util import i18n


class NumberPicker(View):
    def __init__(self, master=None, label_text="no_text", default: float = 0, min_value: float = 0,
                 max_value: float = 100, increment: float = 1, is_float: bool = False,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self.is_float = is_float
        self._label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.spinbox: Union[CustomSpinbox, None] = None
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment
        self.update_language()

    def create_view(self):
        CustomLabel(self, textvariable=self._label_info).pack(side=tk.LEFT)
        self.spinbox = CustomSpinbox(self, min_value=self.min_value, max_value=self.max_value, step_size=self.increment,
                                     default=self.default, use_float=self.is_float)
        self.spinbox.pack(side=tk.LEFT)

    def get_value(self):
        return self.spinbox.get()

    def update_language(self):
        self._label_info.set(i18n.entry_buttons[self.label_text])
