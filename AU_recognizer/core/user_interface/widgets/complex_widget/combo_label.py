import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class ComboLabel(View):
    def __init__(self, master=None, label_text="no_text", values=None, selected="", state="readonly", **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.update_language()
        self.combobox = None
        self.combo_values = values
        self.selected = selected
        self.state = state

    def create_view(self):
        ttk.Label(self, textvariable=self._path_label_info).pack(side=tk.LEFT)
        combo_frame = tk.Frame(self)
        combo_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.combobox = ttk.Combobox(combo_frame, values=self.combo_values, state=self.state)
        self.combobox.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.combobox.set(self.selected)

    def get_value(self):
        return self.combobox.get()

    def update_language(self, values=None, sel=None):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])
        if values is not None:
            self.combo_values = values
            self.combobox['values'] = values
            if sel is not None:
                self.selected = sel
            else:
                self.selected = self.combo_values[0]
            self.combobox.set(self.selected)  # Set to first item by default

    def bind_combobox_event(self, callback):
        """Bind an external function to the Combobox selection event."""
        self.combobox.bind("<<ComboboxSelected>>", lambda event: callback(self.combobox.get()))
