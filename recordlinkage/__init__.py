from recordlinkage.indexing import *
from recordlinkage.comparing import *
from recordlinkage.classifiers import *
from recordlinkage.measures import *
from recordlinkage.utils import split_index

from recordlinkage import rl_logging as logging

from recordlinkage._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ['datasets', 'preprocessing']
