from AU_recognizer.core.util import asset
from .custom_font import CustomFont
from .font_manager import FontManager

FontManager.init_font_manager()

# load Roboto fonts (used on Windows/Linux)
font_directory = asset(asset_name="fonts")
FontManager.load_font(font_directory / "Roboto" / "Roboto-Regular.ttf")
FontManager.load_font(font_directory / "Roboto" / "Roboto-Medium.ttf")
FontManager.load_font(font_directory / "CustomTkinter_shapes_font.otf")
