
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top

from recordlinkage.indexing import *
from recordlinkage.comparing import *
from recordlinkage.classifiers import *
from recordlinkage.measures import *

from recordlinkage import rl_logging as logging

# # import standardise
# from recordlinkage.standardise.cleaning import *
# from recordlinkage.standardise.encoding import *

from recordlinkage._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ['datasets']
