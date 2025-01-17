# import manager classes
from .views.widgets.appearance import AppearanceModeTracker
from .views.widgets.font import FontManager
from .views.widgets.scaling import ScalingTracker
from .views.widgets.theme import ThemeManager
from .views.widgets.core_rendering import DrawEngine

# import base widgets
from .views.widgets.core_rendering import CustomCanvas
from .views.widgets.core_widget_classes import CustomTKBaseClass

# import widgets
from .views.widgets.core_widget_classes import CustomButton
from .views.widgets.core_widget_classes import CustomCheckBox
from .views.widgets.core_widget_classes import CustomComboBox
from .views.widgets.core_widget_classes import CustomEntry
from .views.widgets.core_widget_classes import CustomFrame
from .views.widgets.core_widget_classes import CustomLabel
from .views.widgets.core_widget_classes import CustomOptionMenu
from .views.widgets.core_widget_classes import CustomProgressBar
from .views.widgets.core_widget_classes import CustomRadioButton
from .views.widgets.core_widget_classes import CustomScrollbar
from .views.widgets.core_widget_classes import CustomSegmentedButton
from .views.widgets.core_widget_classes import CustomSlider
from .views.widgets.core_widget_classes import CustomSwitch
from .views.widgets.core_widget_classes import CustomTabview
from .views.widgets.core_widget_classes import CustomTextbox
from .views.widgets.core_widget_classes import ScrollableFrame

# import windows
from .views import CustomTk
from .views import CustomToplevel
from .views import CustomInputDialog

# import font classes
from .views.widgets.font import CustomFont

# import image classes
from .views.widgets.image import CustomTkImage


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
