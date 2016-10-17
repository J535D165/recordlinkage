from __future__ import division
# from __future__ import absolute_import
# from __future__ import unicode_literals

import warnings
import os
import sys

import numpy as np
import pandas

from .cleaning import *
from recordlinkage.comparing import _import_jellyfish


def phonetic(s, method, encoding='utf-8', decode_error='strict'):
    """
    Phonetically encode the values in the Series.

    :param method: The algorithm that is used to phonetically encode the
            values. The possible options are 'soundex'
            (`wikipedia <https://en.wikipedia.org/wiki/Soundex>`_) and
            'nysiis' (`wikipedia <https://en.wikipedia.org/wiki/New_York_State_Identification_and_Intelligence_System>`_).
    :param encoding: If bytes are given, this encoding is used to
            decode.
    :param decode_error: Instruction on what to do if a byte Series is given
            that contains characters not of the given `encoding`. By default,
            it is 'strict', meaning that a UnicodeDecodeError will be raised.
            Other values are 'ignore' and 'replace'.    :type method: str

    :return: A Series with phonetic encoded values.
    :rtype: pandas.Series
    :type encoding: string, default='utf-8'
    :type decode_error: {'strict', 'ignore', 'replace'}

    .. note::

        The 'soundex' and 'nysiis' algorithms use the package 'jellyfish'.
        It can be installed with pip (``pip install jellyfish``).

    """

    # import jellyfish if present
    jellyfish = _import_jellyfish()

    # encoding
    if sys.version_info[0] == 2:
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if type(x) == bytes else x)

    s = s.str.replace('[\-\_\s]', '')

    if method == 'soundex':
        phonetic_callback = jellyfish.soundex
    elif method == 'nysiis':
        phonetic_callback = jellyfish.nysiis
    else:
        raise Exception("Phonetic encoding method not found")

    return s.str.upper().apply(
        lambda x: phonetic_callback(x) if pandas.notnull(x) else np.nan
    )


