# __all__ = ["I will get rewritten"]
## Don't modify the line above, or this line!
# import automodinit
# automodinit.automodinit(__name__, __file__, globals())
# del automodinit

from ghostbox.api import *
from ghostbox.definitions import *
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_ghostbox_data(path):
    """Returns PATH preceded by the location of the ghostbox data dir, which is part of the python site-package."""
    return os.path.join(_ROOT, "data", path)


def get_ghostbox_html_path():
    """Returns the path to the index.html and javascript files for the integrated HTTP server."""
    return os.path.join(_ROOT, "html")



__all__ = [
    "from_generic",
    "from_openai_legacy",
    "from_llamacpp",
    "from_openai_official",
    "from_google",
    "Ghostbox",
    "ChatMessage",
    "LLMBackend",
    "TTSModel",
    "TTSOutputMethod",
    "PromptFormatTemplateSpecialValue",
    "Timings",
    "api_default_options",
    "get_ghostbox_data",
    "get_ghostbox_html_path"
]
