import tkinter as tk
from typing import Union, Literal

from AU_recognizer.core.user_interface import CustomLabel, CustomComboBox
from AU_recognizer.core.user_interface.views import View
from AU_recognizer.core.util import i18n


class ComboLabel(View):
    def __init__(self, master=None, label_text="no_text", values=None, selected="",
                 state: Literal["normal", "disabled", "readonly"] = "readonly", **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.update_language()
        self.combobox: Union[CustomComboBox, None] = None
        self.combo_values = values
        self.selected = selected
        self.state = state

    def create_view(self):
        CustomLabel(self, textvariable=self._path_label_info).pack(side=tk.LEFT, padx=20, pady=(10, 10))
        self.combobox = CustomComboBox(self, values=self.combo_values, state=self.state)
        self.combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=20, pady=(10, 10))
        self.combobox.set(self.selected)

    def get_value(self):
        return self.combobox.get()

    def update_language(self, values=None, sel=None):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])
        if values is not None:
            self.combo_values = values
            self.combobox.configure(values=values)
            if sel is not None:
                self.selected = sel
            else:
                self.selected = self.combo_values[0]
            self.combobox.set(self.selected)  # Set to first item by default

    def bind_combobox_event(self, callback):
        """Bind an external function to the Combobox selection event."""
        self.combobox.bind("<<ComboboxSelected>>", lambda event: callback(self.combobox.get()))
