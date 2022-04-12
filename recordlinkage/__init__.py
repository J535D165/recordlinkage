# register the configuration
import recordlinkage.config_init  # noqa
from recordlinkage import rl_logging as logging
from recordlinkage._version import get_versions
from recordlinkage.annotation import read_annotation_file
from recordlinkage.annotation import write_annotation_file
from recordlinkage.api import Compare
from recordlinkage.api import Index
from recordlinkage.classifiers import *
from recordlinkage.config import describe_option
from recordlinkage.config import get_option
from recordlinkage.config import option_context
from recordlinkage.config import options
from recordlinkage.config import reset_option
from recordlinkage.config import set_option
from recordlinkage.measures import *
from recordlinkage.network import *
from recordlinkage.utils import index_split
from recordlinkage.utils import split_index

__version__ = get_versions()['version']
del get_versions
