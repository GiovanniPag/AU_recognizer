import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

from AU_recognizer.core.util import logger


class FontManager:
    linux_font_paths = [
        Path.home() / ".fonts/",  # Default path
        Path.home() / ".local/share/fonts/",  # Fallback path
    ]

    @classmethod
    def init_font_manager(cls):
        # Linux
        if sys.platform.startswith("linux"):
            try:
                for path in cls.linux_font_paths:
                    if not path.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
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
        fr_private = 0x10
        fr_not_enum = 0x20
        if isinstance(font_path, bytes):
            path_buffer = create_string_buffer(font_path)
            add_font_resource_ex = windll.gdi32.AddFontResourceExA
        elif isinstance(font_path, str):
            path_buffer = create_unicode_buffer(font_path)
            add_font_resource_ex = windll.gdi32.AddFontResourceExW
        else:
            raise TypeError('font_path must be of type bytes or str')
        flags = (fr_private if private else 0) | (fr_not_enum if not enumerable else 0)
        num_fonts_added = add_font_resource_ex(byref(path_buffer), flags, 0)
        return bool(min(num_fonts_added, 1))

    @classmethod
    def load_font(cls, font_path: Union[str, Path]) -> bool:
        """
        Load a font into the system for different platforms.
        """
        # Check if the font file exists
        font_path = Path(font_path)
        if not font_path.is_file():
            logger.error(f"FontManager error: Font file '{font_path}' does not exist.\n")
            return False
        # Windows
        if sys.platform.startswith("win"):
            return cls.windows_load_font(font_path.as_posix(), private=True, enumerable=False)
        # Linux
        elif sys.platform.startswith("linux"):
            for path in cls.linux_font_paths:
                try:
                    dest_path = path / font_path.name
                    if not dest_path.is_file():
                        shutil.copy(font_path, dest_path)
                        cls.refresh_font_cache(path.as_posix())  # Refresh the font cache
                    return True
                except Exception as err:
                    logger.error(f"FontManager error: {str(err)}")
                    return False
        # macOS and others
        else:
            return False

    @staticmethod
    def refresh_font_cache(directory: str):
        """
        Refresh the font cache on Linux using fc-cache.
        """
        try:
            subprocess.run(["fc-cache", "-fv", directory], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as err:
            logger.error(f"FontManager error (fc-cache): {err}\n")
