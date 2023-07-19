# register the configuration
import recordlinkage.config_init  # noqa
from recordlinkage.classifiers import *
from recordlinkage.measures import *
from recordlinkage.network import *


from recordlinkage import rl_logging as logging
from recordlinkage.annotation import read_annotation_file
from recordlinkage.annotation import write_annotation_file
from recordlinkage.api import Compare
from recordlinkage.api import Index
from recordlinkage.config import describe_option
from recordlinkage.config import get_option
from recordlinkage.config import option_context
from recordlinkage.config import options
from recordlinkage.config import reset_option
from recordlinkage.config import set_option
from recordlinkage.utils import index_split
from recordlinkage.utils import split_index

try:
    from recordlinkage._version import __version__
    from recordlinkage._version import __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)


__all__ = [
    "logging",
    "read_annotation_file",
    "write_annotation_file",
    "Compare",
    "Index",
    "describe_option",
    "get_option",
    "option_context",
    "options",
    "reset_option",
    "set_option",
    "index_split",
    "split_index",
]
