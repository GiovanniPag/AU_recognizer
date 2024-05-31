import platform
import re
import tkinter as tk
import tkinter.ttk as ttk
from abc import abstractmethod
from enum import Enum, auto
from math import floor
from pathlib import Path
from tkinter import filedialog

from PIL import ImageTk
from PIL import Image

from AU_recognizer import i18n, I18N_TITLE, RADIO_BTN, RADIO_TEXT
from AU_recognizer.core.util import logger, asset, get_desktop_path


class View(ttk.Frame):
    @abstractmethod
    def create_view(self):
        raise NotImplementedError

    @abstractmethod
    def update_language(self):
        raise NotImplementedError


def check_name(new_val):
    logger.debug(f"validate project name {new_val}")
    return re.match('^[a-zA-Z0-9-_]*$', new_val) is not None and len(new_val) <= 50


def check_file_name(new_val):
    logger.debug(f"validate file name {new_val}")
    return re.match(r'^[a-zA-Z0-9-_\\.]*$', new_val) is not None


def check_num(new_val):
    logger.warning(f"validate only int {new_val}")
    return re.match('^[0-9]*$', new_val) is not None and len(new_val) <= 3


def sizeof_fmt(num, suffix="B"):
    logger.debug(f"format {num} in bytes")
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


class GeometryManager(Enum):
    GRID = auto()
    PACK = auto()


class AutoScrollbar(ttk.Scrollbar):
    def __init__(self, master, geometry: GeometryManager = GeometryManager.GRID, column_grid=1, row_grid=0,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.geometry = geometry
        self.column = column_grid
        self.row = row_grid

    """Create a scrollbar that hides itself if it's not needed."""

    def set(self, lo, hi):
        logger.debug(f"hide scrollbar if not needed")
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            if self.geometry is GeometryManager.GRID:
                self.grid_forget()
            else:
                self.pack_forget()
        else:
            if str(self.cget("orient")) == tk.HORIZONTAL:
                if self.geometry is GeometryManager.GRID:
                    self.grid(column=self.column, row=self.row, sticky=(tk.W, tk.E))
                else:
                    self.pack(fill=tk.X, side=tk.BOTTOM)
            else:
                if self.geometry is GeometryManager.GRID:
                    self.grid(column=self.column, row=self.row, sticky=(tk.N, tk.S))
                else:
                    self.pack(fill=tk.Y, side=tk.RIGHT)
        ttk.Scrollbar.set(self, lo, hi)

    def place(self, **kw):
        raise tk.TclError("cannot use place with this widget")


class DiscreteStep(tk.Scale):
    def __init__(self, master=None, step=1, **kw):
        super().__init__(master, **kw)
        self.step = floor(step)
        self.variable: tk.IntVar = kw.get("variable")
        self.value_list = list(range(int(kw.get("from_")), int(kw.get("to") + 1), self.step))
        self.configure(command=self.value_check)

    def value_check(self, value):
        new_value = min(self.value_list, key=lambda x: abs(x - float(value)))
        self.variable.set(value=new_value)


class AutoWrapMessage(tk.Message):
    def __init__(self, master, margin=8, **kwargs):
        super().__init__(master, **kwargs)
        self.margin = margin
        self.bind("<Configure>", lambda event: event.widget.configure(width=event.width - self.margin))


class ScrollFrame(tk.Frame):
    def __init__(self, master, debug=False):
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.viewPort = tk.Frame(self.canvas)
        if debug:
            self.viewPort.configure(background="#bbaaee")
        self.vsb = AutoScrollbar(self, geometry=GeometryManager.GRID, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.viewPort, anchor="nw", tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.on_frame_configure)
        # bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        # bind an event whenever the size of the canvas frame changes.
        self.viewPort.bind('<Enter>', self.on_enter)
        # bind wheel events when the cursor enters the control
        self.viewPort.bind('<Leave>', self.on_leave)
        # unbind wheel events when the cursor leaves the control
        self.on_frame_configure(None)
        # perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize

    def on_frame_configure(self, _):
        """ Reset the scroll region to encompass the inner frame """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # whenever the size of the frame changes, alter the scroll region respectively.

    def on_canvas_configure(self, event):
        """ Reset the canvas window to encompass inner frame when required """
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        # whenever the size of the canvas changes alter the window region respectively.

    def on_mouse_wheel(self, event):  # cross platform scroll wheel event
        if platform.system() == 'Windows':
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif platform.system() == 'Darwin':
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def on_enter(self, event):
        # bind wheel events when the cursor enters the control
        if self.vsb.winfo_ismapped():
            if platform.system() == 'Linux':
                self.canvas.bind_all("<Button-4>", self.on_mouse_wheel)
                self.canvas.bind_all("<Button-5>", self.on_mouse_wheel)
            else:
                self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

    def on_leave(self, event):
        # unbind wheel events when the cursor leaves the control
        if platform.system() == 'Linux':
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        else:
            self.canvas.unbind_all("<MouseWheel>")


class ToolTip:
    """
    create a tooltip for a given widget
    """

    def __init__(self, widget, bg='#FFFFEA', pad=(5, 3, 5, 3), text='widget info',
                 wait_time=400, wrap_length=250):
        self.wait_time = wait_time  # milliseconds
        self.wrap_length = wrap_length  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None

    def on_enter(self, _=None):
        self.schedule()

    def on_leave(self, _=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.wait_time, self.showtip)

    def unschedule(self):
        old_id = self.id
        self.id = None
        if old_id:
            self.widget.after_cancel(old_id)

    @staticmethod
    def __tip_pos_calculator(widget_, label_, tip_delta=(10, 5), pad_=(5, 3, 5, 3)):
        s_width, s_height = widget_.winfo_screenwidth(), widget_.winfo_screenheight()
        width, height = (pad_[0] + label_.winfo_reqwidth() + pad_[2],
                         pad_[1] + label_.winfo_reqheight() + pad_[3])
        mouse_x, mouse_y = widget_.winfo_pointerxy()
        x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
        x2, y2 = x1 + width, y1 + height
        x_delta = x2 - s_width
        if x_delta < 0:
            x_delta = 0
        y_delta = y2 - s_height
        if y_delta < 0:
            y_delta = 0
        if (x_delta, y_delta) != (0, 0):
            if x_delta:
                x1 = mouse_x - tip_delta[0] - width
            if y_delta:
                y1 = mouse_y - tip_delta[1] - height
        if y1 < 0:  # out on the top
            y1 = 0
        return x1, y1

    def showtip(self):
        # creates a top level window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        win = tk.Frame(self.tw, background=self.bg, borderwidth=0)
        label = ttk.Label(win, text=self.text, justify=tk.LEFT, background=self.bg,
                          relief=tk.SOLID, borderwidth=0, wraplength=self.wrap_length)
        label.grid(padx=(self.pad[0], self.pad[2]), pady=(self.pad[1], self.pad[3]), sticky=tk.NSEW)
        win.grid()
        x, y = self.__tip_pos_calculator(self.widget, label)
        self.tw.wm_geometry("+%d+%d" % (x, y))

    def hidetip(self):
        old_tw = self.tw
        self.tw = None
        if old_tw:
            old_tw.destroy()


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


class ComboEntry(View):
    def __init__(self, master=None, label_text="no_text", values="", selected="", state="readonly", **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.update_language()
        self.languages_combobox = None
        self.combo_values = values
        self.selected = selected
        self.state = state

    def create_view(self):
        ttk.Label(self, textvariable=self._path_label_info).pack(side=tk.LEFT)
        combo_frame = tk.Frame(self)
        combo_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.languages_combobox = ttk.Combobox(combo_frame, values=self.combo_values, state=self.state)
        self.languages_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.languages_combobox.set(self.selected)

    def get_value(self):
        return self.languages_combobox.get()

    def update_language(self):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])


class CheckLabel(View):
    def __init__(self, master=None, label_text="no_text", default=False, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.label_text = label_text
        self._path_label_info = tk.StringVar(value=i18n.entry_buttons[label_text])
        self.value = tk.BooleanVar(value=default)
        self.update_language()

    def create_view(self):
        check_box = ttk.Checkbutton(self, textvariable=self._path_label_info, variable=self.value)
        check_box.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def get_value(self):
        return self.value.get()

    def update_language(self):
        self._path_label_info.set(i18n.entry_buttons[self.label_text])


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
