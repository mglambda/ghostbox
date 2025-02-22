#__all__ = ["I will get rewritten"]
## Don't modify the line above, or this line!
#import automodinit
#automodinit.automodinit(__name__, __file__, globals())
#del automodinit

from ghostbox.api import *

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_ghostbox_data(path):
    return os.path.join(_ROOT, 'data', path)
