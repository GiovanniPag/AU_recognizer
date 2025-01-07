import tkinter as tk
from tkinter import ttk as ttk

from AU_recognizer.core.views import View


class RadioList(View):
    def __init__(self, master=None, list_title="no_text", default="", data=None, orientation=tk.HORIZONTAL, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.list_title = list_title
        self.selected_mode = tk.StringVar(value=default)
        self.radio_frame = ttk.LabelFrame(self, text=i18n.entry_buttons[self.list_title])
        self.radio_buttons = {}
        self.orientation = orientation
        for btn_name in data:
            text = tk.StringVar(value=i18n.radio_buttons[btn_name])
            self.radio_buttons[btn_name] = {
                RADIO_TEXT: text,
                RADIO_BTN: ttk.Radiobutton(master=self.radio_frame, textvariable=text, variable=self.selected_mode,
                                           value=btn_name)
            }
        self.update_language()

    def create_view(self):
        self.radio_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        for btn_name, btn_data in self.radio_buttons.items():
            btn = btn_data[RADIO_BTN]
            if self.orientation == tk.HORIZONTAL:
                btn.pack(side=tk.LEFT, padx=5, pady=5)
            elif self.orientation == tk.VERTICAL:
                btn.pack(anchor=tk.W, padx=5, pady=5)

    def get_value(self):
        return self.selected_mode.get()

    def update_language(self):
        self.radio_frame.configure(text=i18n.entry_buttons[self.list_title])
        for btn_name, btn_data in self.radio_buttons.items():
            btn_data[RADIO_TEXT].set(value=i18n.radio_buttons[btn_name])
