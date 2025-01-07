import tkinter as tk
from pathlib import Path
from tkinter import ttk as ttk, filedialog

from PIL import Image, ImageTk

from AU_recognizer.core.util import get_desktop_path, asset, logger
from AU_recognizer.core.views import View


class EntryButton(View):
    def __init__(self, master=None, label_text="general", entry_text="", **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.cwd = entry_text if Path(entry_text).is_dir() else get_desktop_path()
        self._path_label = tk.StringVar(value=entry_text.strip())
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.update_language()
        # Load folder icon
        self.folder_icon = Image.open(asset("folder_icon.png"))
        self.folder_icon = self.folder_icon.resize((24, 24))
        self.folder_icon = ImageTk.PhotoImage(self.folder_icon)

    def create_view(self):
        ttk.Label(self, textvariable=self._path_label_info).pack(side=tk.LEFT)
        entry_frame = tk.Frame(self)
        entry_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        entry = ttk.Entry(entry_frame, textvariable=self._path_label)
        entry.config(width=len(entry.get()))
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        folder_button = tk.Button(entry_frame, image=self.folder_icon, command=self.browse_folder, bd=0)
        folder_button.pack(side=tk.RIGHT, padx=5)

    def get_value(self):
        return self._path_label.get()

    def update_language(self):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])

    def browse_folder(self):
        logger.debug("choose directory")
        dir_name = filedialog.askdirectory(initialdir=self.cwd, mustexist=True,
                                           parent=self.master, title=i18n.choose_folder_dialog_g[I18N_TITLE])
        if dir_name != () and dir_name:
            self._path_label.set(dir_name)
