import tkinter
import tkinter.ttk as ttk
from typing import Union, Callable, Tuple, Any

from AU_recognizer.core.util import pop_from_dict_by_set, check_kwargs_empty, logger
from .. import core_widget_classes
from ... import CustomTk, CustomToplevel

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from ..theme import ThemeManager
from ..font import CustomFont
from ..image import CustomTkImage
from ..appearance import CustomAppearanceModeBaseClass
from ..scaling import CustomScalingBaseClass


class CustomTKBaseClass(tkinter.Frame, CustomAppearanceModeBaseClass, CustomScalingBaseClass):
    """ Base class of every widget, handles the dimensions, bg_color,
        appearance_mode changes, scaling, bg changes of master if master is not a custom widget """
    # attributes that are passed to and managed by the tkinter frame only:
    _valid_tk_frame_attributes: set = {"cursor"}
    _cursor_manipulation_enabled: bool = True

    def __init__(self,
                 master: Any,
                 width: int = 0,
                 height: int = 0,
                 bg_color: Union[str, Tuple[str, str]] = "transparent",
                 **kwargs):
        # call init methods of super classes
        tkinter.Frame.__init__(self, master=master, width=width, height=height,
                               **pop_from_dict_by_set(kwargs, self._valid_tk_frame_attributes))
        CustomAppearanceModeBaseClass.__init__(self)
        CustomScalingBaseClass.__init__(self, scaling_type="widget")
        # check if kwargs is empty, if not raise error for unsupported arguments
        check_kwargs_empty(kwargs, raise_error=True)
        # dimensions independent of scaling, _current_width and _current_height in pixel,
        # represent current size of the widget, _current_width and _current_height are independent of the scale
        self._current_width = width
        self._current_height = height
        # _desired_width and _desired_height, represent desired size set by width and height
        self._desired_width = width
        self._desired_height = height
        # set width and height of tkinter.Frame
        super().configure(width=self._apply_widget_scaling(self._desired_width),
                          height=self._apply_widget_scaling(self._desired_height))

        # save latest geometry function and kwargs
        class GeometryCallDict(TypedDict):
            function: Callable
            kwargs: dict

        self._last_geometry_manager_call: Union[GeometryCallDict, None] = None

        # background color
        self._bg_color: Union[str, Tuple[
            str, str]] = self._detect_color_of_master() if bg_color == "transparent" else self._check_color_type(
            bg_color, transparency=True)

        # set bg color of tkinter.Frame
        super().configure(bg=self._apply_appearance_mode(self._bg_color))

        # add configure callback to tkinter.Frame
        super().bind('<Configure>', self._update_dimensions_event)

        # overwrite configure methods of master when master is tkinter widget,
        # so that bg changes get applied on child widget as well
        if isinstance(self.master, (
                tkinter.Tk, tkinter.Toplevel, tkinter.Frame, tkinter.LabelFrame, ttk.Frame, ttk.LabelFrame,
                ttk.Notebook)) and not isinstance(self.master, (CustomTKBaseClass, CustomAppearanceModeBaseClass)):
            master_old_configure = self.master.config

            def new_configure(*args, **inner_kwargs):
                if "bg" in inner_kwargs:
                    self.configure(bg_color=inner_kwargs["bg"])
                elif "background" in inner_kwargs:
                    self.configure(bg_color=inner_kwargs["background"])

                # args[0] is dict when attribute gets changed by widget[<attribute>] syntax
                elif len(args) > 0 and type(args[0]) is dict:
                    if "bg" in args[0]:
                        self.configure(bg_color=args[0]["bg"])
                    elif "background" in args[0]:
                        self.configure(bg_color=args[0]["background"])
                master_old_configure(*args, **inner_kwargs)

            self.master.config = new_configure
            self.master.configure = new_configure

    def destroy(self):
        """ Destroy this and all descendants widgets. """
        # call destroy methods of super classes
        tkinter.Frame.destroy(self)
        CustomAppearanceModeBaseClass.destroy(self)
        CustomScalingBaseClass.destroy(self)

    def _draw(self, no_color_updates: bool = False):
        """ can be overridden but super method must be called """
        if no_color_updates is False:
            pass

    def config(self, *args, **kwargs):
        raise AttributeError(
            "'config' is not implemented for custom widgets. For consistency, always use 'configure' instead.")

    def configure(self, require_redraw=False, **kwargs):
        """ basic configure with bg_color, width, height support, calls configure of tkinter.Frame, updates in the
        end"""

        if "width" in kwargs:
            self._set_dimensions(width=kwargs.pop("width"))

        if "height" in kwargs:
            self._set_dimensions(height=kwargs.pop("height"))

        if "bg_color" in kwargs:
            new_bg_color = self._check_color_type(kwargs.pop("bg_color"), transparency=True)
            if new_bg_color == "transparent":
                self._bg_color = self._detect_color_of_master()
            else:
                self._bg_color = self._check_color_type(new_bg_color)
            require_redraw = True

        super().configure(**pop_from_dict_by_set(kwargs, self._valid_tk_frame_attributes))  # configure tkinter.Frame

        # if there are still items in the kwargs dict, raise ValueError
        check_kwargs_empty(kwargs, raise_error=True)

        if require_redraw:
            self._draw()

    def cget(self, attribute_name: str):
        """ basic cget with bg_color, width, height support, calls cget of tkinter.Frame """

        if attribute_name == "bg_color":
            return self._bg_color
        elif attribute_name == "width":
            return self._desired_width
        elif attribute_name == "height":
            return self._desired_height

        elif attribute_name in self._valid_tk_frame_attributes:
            return super().cget(attribute_name)  # cget of tkinter.Frame
        else:
            raise ValueError(
                f"'{attribute_name}' is not a supported argument. Look at the documentation for supported arguments.")

    def _check_font_type(self, font: any):
        """ check font type when passed to widget """
        if isinstance(font, CustomFont):
            return font
        elif type(font) is tuple and len(font) == 1:
            logger.warn(
                f"{type(self).__name__} Warning: font {font} given without size, will be extended with default text "
                f"size of current theme\n")
            return font[0], ThemeManager.theme["text"]["size"]
        elif type(font) is tuple and 2 <= len(font) <= 6:
            return font
        else:
            raise ValueError(f"Wrong font type {type(font)}\n" +
                             f"For consistency, Customtkinter requires the font argument to be a tuple of len 2 to 6 "
                             f"or an instance of CustomFont.\n" +
                             f"\nUsage example:\n" +
                             f"font=customtkinter.CustomFont(family='<name>', size=<size in px>)\n" +
                             f"font=('<name>', <size in px>)\n")

    def _check_image_type(self, image: any):
        """ check image type when passed to widget """
        if image is None:
            return image
        elif isinstance(image, CustomTkImage):
            return image
        else:
            logger.warn(
                f"{type(self).__name__} Warning: Given image is not CTkImage but {type(image)}. Image can not be "
                f"scaled on HighDPI displays, use CustomTkImage instead.\n")
            return image

    def _update_dimensions_event(self, event):
        # only redraw if dimensions changed (for performance), independent of scaling
        if round(self._current_width) != round(self._reverse_widget_scaling(event.width)) or round(
                self._current_height) != round(self._reverse_widget_scaling(event.height)):
            self._current_width = self._reverse_widget_scaling(
                event.width)  # adjust current size according to new size given by event
            self._current_height = self._reverse_widget_scaling(
                event.height)  # _current_width and _current_height are independent of the scaleZ
            self._draw(no_color_updates=True)  # faster drawing without color changes

    def _detect_color_of_master(self, master_widget=None) -> Union[str, Tuple[str, str]]:
        """ detect foreground color of master widget for bg_color and transparent color """

        if master_widget is None:
            master_widget = self.master

        if isinstance(master_widget, (
                CustomTKBaseClass, CustomTk, CustomToplevel,
                core_widget_classes.scrollable_frame.ScrollableFrame)):
            if master_widget.cget("fg_color") is not None and master_widget.cget("fg_color") != "transparent":
                return master_widget.cget("fg_color")
            elif isinstance(master_widget, core_widget_classes.scrollable_frame.ScrollableFrame):
                return self._detect_color_of_master(master_widget.master.master.master)
            # if fg_color of master is None, try to retrieve fg_color from master of master
            elif hasattr(master_widget, "master"):
                return self._detect_color_of_master(master_widget.master)
        elif isinstance(master_widget, (ttk.Frame, ttk.LabelFrame, ttk.Notebook, ttk.Label)):  # master is ttk widget
            try:
                ttk_style = ttk.Style()
                return ttk_style.lookup(master_widget.winfo_class(), 'background')
            except Exception as err:
                logger.error(err)
                return "#FFFFFF", "#000000"
        else:  # master is normal tkinter widget
            try:
                return master_widget.cget("bg")  # try to get bg color by .cget() method
            except Exception as err:
                logger.error(err)
                return "#FFFFFF", "#000000"

    def _set_appearance_mode(self, mode_string):
        super()._set_appearance_mode(mode_string)
        self._draw()
        super().update_idletasks()

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        super()._set_scaling(new_widget_scaling, new_window_scaling)
        super().configure(width=self._apply_widget_scaling(self._desired_width),
                          height=self._apply_widget_scaling(self._desired_height))
        if self._last_geometry_manager_call is not None:
            self._last_geometry_manager_call["function"](
                **self._apply_argument_scaling(self._last_geometry_manager_call["kwargs"]))

    def _set_dimensions(self, width=None, height=None):
        if width is not None:
            self._desired_width = width
        if height is not None:
            self._desired_height = height
        super().configure(width=self._apply_widget_scaling(self._desired_width),
                          height=self._apply_widget_scaling(self._desired_height))

    def bind(self, sequence=None, command=None, add=None):
        raise NotImplementedError

    def unbind(self, sequence=None, funcid=None):
        raise NotImplementedError

    def unbind_all(self, sequence):
        raise AttributeError(
            "'unbind_all' is not allowed, because it would delete necessary internal callbacks for all widgets")

    def bind_all(self, sequence=None, func=None, add=None):
        raise AttributeError("'bind_all' is not allowed, could result in undefined behavior")

    def place(self, **kwargs):
        """
        Place a widget in the parent widget. Use as options:
        in=master - master relative to which the widget is placed
        in_=master - see 'in' option description
        x=amount - locate anchor of this widget at position x of master
        y=amount - locate anchor of this widget at position y of master
        relx=amount - locate anchor of this widget between 0.0 and 1.0 relative to width of master (1.0 is right edge)
        rely=amount - locate anchor of this widget between 0.0 and 1.0 relative to height of master (1.0 is bottom edge)
        anchor=NSEW (or subset) - position anchor according to given direction
        width=amount - width of this widget in pixel
        height=amount - height of this widget in pixel
        relwidth=amount - width of this widget between 0.0 and 1.0 relative to width of master (1.0 is the same width as
        the master)
        relheight=amount - height of this widget between 0.0 and 1.0 relative to height of master (1.0 is the same
        height as the master)
        bordermode="inside" or "outside" - whether to take border width of master widget into account
        """
        if "width" in kwargs or "height" in kwargs:
            raise ValueError(
                "'width' and 'height' arguments must be passed to the constructor of the widget, not the place method")
        self._last_geometry_manager_call = {"function": super().place, "kwargs": kwargs}
        return super().place(**self._apply_argument_scaling(kwargs))

    def place_forget(self):
        """ Unmap this widget. """
        self._last_geometry_manager_call = None
        return super().place_forget()

    def pack(self, **kwargs):
        """
        Pack a widget in the parent widget. Use as options:
        after=widget - pack it after you have packed widget
        anchor=NSEW (or subset) - position widget according to given direction
        before=widget - pack it before you will pack widget
        expand=bool - expand widget if parent size grows
        fill=NONE or X or Y or BOTH - fill widget if widget grows
        in=master - use master to contain this widget
        in_=master - see 'in' option description
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        side=TOP or BOTTOM or LEFT or RIGHT -  where to add this widget.
        """
        self._last_geometry_manager_call = {"function": super().pack, "kwargs": kwargs}
        return super().pack(**self._apply_argument_scaling(kwargs))

    def pack_forget(self):
        """ Unmap this widget and do not use it for the packing order. """
        self._last_geometry_manager_call = None
        return super().pack_forget()

    def grid(self, **kwargs):
        """
        Position a widget in the parent widget in a grid. Use as options:
        column=number - use cell identified with given column (starting with 0)
        columnspan=number - this widget will span several columns
        in=master - use master to contain this widget
        in_=master - see 'in' option description
        ipadx=amount - add internal padding in x direction
        ipady=amount - add internal padding in y direction
        padx=amount - add padding in x direction
        pady=amount - add padding in y direction
        row=number - use cell identified with given row (starting with 0)
        rowspan=number - this widget will span several rows
        sticky=NSEW - if cell is larger on which sides will this widget stick to the cell boundary
        """
        self._last_geometry_manager_call = {"function": super().grid, "kwargs": kwargs}
        return super().grid(**self._apply_argument_scaling(kwargs))

    def grid_forget(self):
        """ Unmap this widget. """
        self._last_geometry_manager_call = None
        return super().grid_forget()
