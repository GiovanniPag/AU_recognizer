# import manager classes
from .widgets.appearance import AppearanceModeTracker
# import font classes
from .widgets.font import CustomFont
from .widgets.font import FontManager
from .widgets.scaling import ScalingTracker
from .widgets.theme import ThemeManager
# import base widgets
from .widgets.core_rendering import DrawEngine
from .widgets.core_rendering import CustomCanvas
# import windows
from .custom_tk import CustomTk
from .custom_toplevel import CustomToplevel
from .widgets.core_widget_classes import CustomTKBaseClass
# import image classes
from .widgets.image import CustomTkImage
# import widgets
from .widgets.core_widget_classes import CustomButton
from .widgets.core_widget_classes import CustomCheckBox
from .widgets.core_widget_classes import CustomComboBox
from .widgets.core_widget_classes import CustomEntry
from .widgets.core_widget_classes import CustomFrame
from .widgets.core_widget_classes import CustomLabel
from .widgets.core_widget_classes import CustomOptionMenu
from .widgets.core_widget_classes import CustomProgressBar
from .widgets.core_widget_classes import CustomRadioButton
from .widgets.core_widget_classes import CustomScrollbar
from .widgets.core_widget_classes import CustomSegmentedButton
from .widgets.core_widget_classes import CustomSlider
from .widgets.core_widget_classes import CustomSpinbox
from .widgets.core_widget_classes import CustomSwitch
from .widgets.core_widget_classes import CustomTabview
from .widgets.core_widget_classes import CustomTextbox
from .widgets.core_widget_classes import CustomTooltip
from .widgets.core_widget_classes import ScrollableFrame
# dialog
from .dialogs import CustomInputDialog


def set_appearance_mode(mode_string: str):
    """ possible values: light, dark, system """
    AppearanceModeTracker.set_appearance_mode(mode_string)


def get_appearance_mode() -> str:
    """ get current state of the appearance mode (light or dark) """
    if AppearanceModeTracker.appearance_mode == 0:
        return "Light"
    elif AppearanceModeTracker.appearance_mode == 1:
        return "Dark"


def set_default_color_theme(color_string: str):
    """ set color theme or load custom theme file by passing the path """
    ThemeManager.load_theme(color_string)


def set_widget_scaling(scaling_value: float):
    """ set scaling for the widget dimensions """
    ScalingTracker.set_widget_scaling(scaling_value)


def set_window_scaling(scaling_value: float):
    """ set scaling for window dimensions """
    ScalingTracker.set_window_scaling(scaling_value)
