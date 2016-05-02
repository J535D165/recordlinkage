
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Import important modules
from recordlinkage.standardise import *

from .indexing import *
from .comparing import *
from .classifier import *
from .measures import *



