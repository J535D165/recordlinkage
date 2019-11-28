
# register the configuration
import recordlinkage.config_init  # noqa

from recordlinkage.api import Compare, Index
from recordlinkage.index import FullIndex
from recordlinkage.index import BlockIndex
from recordlinkage.index import SortedNeighbourhoodIndex
from recordlinkage.index import RandomIndex
from recordlinkage.classifiers import *
from recordlinkage.measures import *
from recordlinkage.network import *
from recordlinkage.utils import split_index, index_split
from recordlinkage.config import (get_option, set_option, reset_option,
                                  describe_option, option_context, options)
from recordlinkage.annotation import read_annotation_file, write_annotation_file
from recordlinkage import rl_logging as logging

from recordlinkage.deprecated import *

from recordlinkage._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ['datasets', 'preprocessing']
