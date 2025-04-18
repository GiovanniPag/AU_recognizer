import sys
from typing import Union, Tuple

from . import CustomLabel
from ... import CustomToplevel, AppearanceModeTracker


class CustomTooltip(CustomToplevel):
    # Mouse hover tooltips that can be attached to widgets
    # noinspection PyMissingConstructor
    def __init__(self, master, text: str = 'Custom Tooltip', delay: int = 400, wrap_length: int = -1,
                 bg_color: Union[str, Tuple[str, str]] = "transparent",
                 fg_color: Union[str, Tuple[str, str]] = "default", mouse_offset: Tuple[int, int] = (1, 1),
                 **kwargs):
        self.wait_time = delay  # milliseconds until tooltip appears
        self.wrap_length = wrap_length  # wrap length of the tooltip text
        self.master = master  # parent widget
        self.text = text  # text to display
        self.mouse_offset = mouse_offset  # offset from mouse position (x, y)
        self.master.bind("<Enter>", self._schedule, add="+")
        self.master.bind("<Leave>", self._leave)
        self.master.bind("<ButtonPress>", self._leave, add="+")
        label = self.master.winfo_children()[0]
        label.bind("<Enter>", self._schedule, add="+")
        self._id = None
        self.kwargs = kwargs
        self._visible = False
        self._is_hovering_tooltip = False
        self._bg_color_is_default = True if bg_color == "transparent" else False
        # used on linux because rounded corners doesn't seem to be possible usually

        # determine colors
        self.__appearance_mode = AppearanceModeTracker.get_mode()
        if fg_color == "default":
            self.fg_color = '#CBCBCB' if self.__appearance_mode == 0 else '#545454'
        else:
            self.fg_color = fg_color
        if bg_color == "transparent":
            if bg_color.startswith('#'):
                color_list = [int(self.fg_color[i:i + 2], 16) for i in range(1, len(self.fg_color), 2)]
                if not any(color == 255 for color in color_list):
                    for i in range(len(color_list)):
                        color_list[i] += 1
                else:
                    for i in range(len(color_list)):
                        color_list[i] -= 1
                self.bg_color = "#" + ''.join(['{:02x}'.format(x) for x in color_list])
            else:
                if self.__appearance_mode == 0:
                    self.bg_color = 'gray86' if self.fg_color != 'gray86' else 'gray84'
                else:
                    self.bg_color = 'gray17' if self.fg_color != 'gray17' else 'gray15'
        else:
            self.bg_color = bg_color

    def _leave(self, _=None):
        self._unschedule()
        if self._visible:
            self.hide()

    def _schedule(self, _=None):
        self._unschedule()
        self._id = self.master.after(self.wait_time, self.show)

    def _unschedule(self):
        # Unschedule scheduled popups
        idx = self._id
        self._id = None
        if idx:
            self.master.after_cancel(idx)

    def show(self, _=None):
        # Get the position the tooltip needs to appear at
        if self._visible:
            return
        super().__init__(self.master)
        super().withdraw()  # hide and reshow window once all code is ran to fix issues due to slower machines (??)
        self._visible = True
        x, y, cx, cy = self.master.bbox("insert")  # type: ignore
        # Has to be offset from mouse position, otherwise it will appear and disappear instantly because it left the
        # parent widget
        x += self.master.winfo_pointerx() + self.mouse_offset[0]
        y += self.master.winfo_pointery() + self.mouse_offset[1]
        if sys.platform.startswith("win"):
            self.wm_attributes('-transparentcolor', self.bg_color)  # used for rounded corners
            self.wm_attributes("-toolwindow", True)  # removes icon from taskbar
            super().configure(bg_color=self.bg_color)
        elif sys.platform == 'darwin':
            self.wm_attributes('-transparent', True)  # used for rounded corners
            super().configure(bg_color='systemTransparent')
        elif sys.platform.startswith("linux"):
            if self._bg_color_is_default:
                self.bg_color = self.fg_color  # create square edge tooltips
        self.wm_overrideredirect(True)
        self.wm_geometry(f'+{x}+{y}')
        label = CustomLabel(self, text=self.text, corner_radius=10, bg_color=self.bg_color, fg_color=self.fg_color,
                            width=1, wraplength=self.wrap_length, **self.kwargs)
        label.pack()
        if sys.platform == 'darwin':
            label.configure(bg_color='systemTransparent')
        label.bind("<Enter>", self._leave, add="+")
        super().deiconify()

    def hide(self):
        self._unschedule()
        self.withdraw()
        self._visible = False

    def configure(self, **kwargs):
        # Change attributes of the tooltip, and redraw if necessary
        require_redraw = False
        if "fg_color" in kwargs:
            self.fg_color = kwargs.pop("fg_color")
            require_redraw = True
        if "bg_color" in kwargs:
            self.bg_color = kwargs.pop("bg_color")
            require_redraw = True
        if "text" in kwargs:
            self.text = kwargs.pop("text")
            require_redraw = True
        if "delay" in kwargs:
            self.wait_time = kwargs.pop("delay")
        if "wrap_length" in kwargs:
            self.wrap_length = kwargs.pop("wrap_length")
            require_redraw = True
        if "mouse_offset" in kwargs:
            self.mouse_offset = kwargs.pop("mouse_offset")
            require_redraw = True
        self.kwargs = kwargs
        if require_redraw:
            self.hide()
            self.show()

    def is_visible(self):
        return self._visible
