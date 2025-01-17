from .theme_manager import ThemeManager

# load default blue theme
try:
    ThemeManager.load_theme("blue")
except FileNotFoundError as err:
    raise FileNotFoundError(f"{err}\nThe .json theme file for CustomTkinter could not be found.\n")
