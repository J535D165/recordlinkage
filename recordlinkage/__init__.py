# register the configuration
import recordlinkage.config_init  # noqa
from recordlinkage.classifiers import *
from recordlinkage.measures import *
from recordlinkage.network import *

try:
    from recordlinkage._version import __version__
    from recordlinkage._version import __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
