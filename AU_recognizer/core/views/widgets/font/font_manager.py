import os
import sys
import shutil
from pathlib import Path
from typing import Union

from AU_recognizer import logger


class FontManager:
    linux_font_path = Path(os.getenv("LINUX_FONT_PATH", "~/.fonts")).expanduser()

    @classmethod
    def init_font_manager(cls):
        # Linux
        if sys.platform.startswith("linux"):
            try:
                if not cls.linux_font_path.is_dir():
                    cls.linux_font_path.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as err:
                logger.error(f"FontManager error: {err}")
                return False
        # other platforms
        else:
            return True

    @classmethod
    def windows_load_font(cls, font_path: Union[str, bytes], private: bool = True,
                          enumerable: bool = False) -> bool:
        from ctypes import windll, byref, create_unicode_buffer, create_string_buffer
        FR_PRIVATE = 0x10
        FR_NOT_ENUM = 0x20
        if isinstance(font_path, bytes):
            path_buffer = create_string_buffer(font_path)
            add_font_resource_ex = windll.gdi32.AddFontResourceExA
        elif isinstance(font_path, str):
            path_buffer = create_unicode_buffer(font_path)
            add_font_resource_ex = windll.gdi32.AddFontResourceExW
        else:
            raise TypeError('font_path must be of type bytes or str')
        flags = (FR_PRIVATE if private else 0) | (FR_NOT_ENUM if not enumerable else 0)
        num_fonts_added = add_font_resource_ex(byref(path_buffer), flags, 0)
        return bool(min(num_fonts_added, 1))

    @classmethod
    def load_font(cls, font_path: Union[str, Path]) -> bool:
        font_path = Path(font_path)
        # Windows
        if sys.platform.startswith("win"):
            return cls.windows_load_font(font_path.as_posix(), private=True, enumerable=False)
        # Linux
        elif sys.platform.startswith("linux"):
            try:
                shutil.copy(font_path, cls.linux_font_path)
                return True
            except Exception as err:
                logger.error(f"FontManager error: {str(err)}")
                return False
        # macOS and others
        else:
            return False
