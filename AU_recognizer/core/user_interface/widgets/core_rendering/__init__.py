import sys

from .custom_canvas import CustomCanvas
from .draw_engine import DrawEngine

CustomCanvas.init_font_character_mapping()

# determine draw method based on current platform
if sys.platform == "darwin":
    DrawEngine.preferred_drawing_method = "polygon_shapes"
else:
    DrawEngine.preferred_drawing_method = "polygon_shapes"
