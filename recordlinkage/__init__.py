
from .indexing import *
from .comparing import *
from .classifiers import *
from .measures import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ['standardise', 'datasets']
