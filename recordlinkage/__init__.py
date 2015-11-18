
# Import important modules
from standardise import *
from indexing import *
from comparing import *

from estimation import *
from classify import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
