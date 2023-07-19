# This module is renamed into preprocessing. Please use the preprocessing
# module instead of this module.

import warnings

from recordlinkage.preprocessing import clean as _clean
from recordlinkage.preprocessing import phonenumbers as _phonenumbers
from recordlinkage.preprocessing import phonetic as _phonetic
from recordlinkage.preprocessing import value_occurence as _value_occurence


def _depr_warn():
    warnings.warn(
        "module recordlinkage.standardise is deprecated, use "
        "recordlinkage.preprocessing instead",
        DeprecationWarning,
        stacklevel=2,
    )


def clean(*args, **kwargs):
    _depr_warn()

    return _clean(*args, **kwargs)


def phonenumbers(*args, **kwargs):
    _depr_warn()

    return _phonenumbers(*args, **kwargs)


def value_occurence(*args, **kwargs):
    _depr_warn()

    return _value_occurence(*args, **kwargs)


def phonetic(*args, **kwargs):
    _depr_warn()

    return _phonetic(*args, **kwargs)
