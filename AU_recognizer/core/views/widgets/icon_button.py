import tkinter as tk

from PIL import Image, ImageTk

from AU_recognizer.core.util import asset
from AU_recognizer.core.views import View, ToolTip


class IconButton(View):
    def __init__(self, master=None, asset_name="", tooltip="", command=..., **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        # Load icon
        self.icon = Image.open(asset(asset_name))
        self.icon = self.icon.resize((24, 24))
        self.icon = ImageTk.PhotoImage(self.icon)
        self.icon_button = tk.Button(self, image=self.icon, command=command, bd=0)
        self.tooltip = tooltip
        if self.tooltip:
            self.tooltip_wid = ToolTip(widget=self.icon_button, text=i18n.tooltips[self.tooltip])

    def create_view(self):
        # Create a button with the folder icon
        self.icon_button.grid(row=0, column=0, sticky="nsew")

    def update_language(self):
        if self.tooltip:
            self.tooltip_wid = ToolTip(widget=self.icon_button, text=i18n.tooltips[self.tooltip])
