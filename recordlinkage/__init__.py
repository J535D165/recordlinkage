# register the configuration
import recordlinkage.config_init  # noqa
from recordlinkage.classifiers import FellegiSunter
from recordlinkage.classifiers import KMeansClassifier
from recordlinkage.classifiers import LogisticRegressionClassifier
from recordlinkage.classifiers import NaiveBayesClassifier
from recordlinkage.classifiers import SVMClassifier
from recordlinkage.classifiers import ECMClassifier
from recordlinkage.measures import reduction_ratio
from recordlinkage.measures import max_pairs
from recordlinkage.measures import full_index_size
from recordlinkage.measures import true_positives
from recordlinkage.measures import true_negatives
from recordlinkage.measures import false_positives
from recordlinkage.measures import false_negatives
from recordlinkage.measures import confusion_matrix
from recordlinkage.measures import precision
from recordlinkage.measures import recall
from recordlinkage.measures import accuracy
from recordlinkage.measures import specificity
from recordlinkage.measures import fscore
from recordlinkage.network import OneToOneLinking
from recordlinkage.network import OneToManyLinking
from recordlinkage.network import ConnectedComponents
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
    "FellegiSunter",
    "KMeansClassifier",
    "LogisticRegressionClassifier",
    "NaiveBayesClassifier",
    "SVMClassifier",
    "ECMClassifier",
    "reduction_ratio",
    "max_pairs",
    "full_index_size",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "confusion_matrix",
    "precision",
    "recall",
    "accuracy",
    "specificity",
    "fscore",
    "OneToOneLinking",
    "OneToManyLinking",
    "ConnectedComponents",
]
