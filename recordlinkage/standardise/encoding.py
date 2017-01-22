import sys

import numpy as np
import pandas

import jellyfish


def phonetic(s, method, concat=True, encoding='utf-8', decode_error='strict'):
    """Convert names or strings into phonetic codes. 

    The implemented algorithms are `soundex
    <https://en.wikipedia.org/wiki/Soundex>`_, `nysiis
    <https://en.wikipedia.org/wiki/New_York_State_Identification_and_
    Intelligence_System>`_, `metaphone
    <https://en.wikipedia.org/wiki/Metaphone>`_ or  `match_rating
    <https://en.wikipedia.org/wiki/Match_rating_approach>`_.

    Parameters
    ----------
    method: string
        The algorithm that is used to phonetically encode the values. The
        possible options are "soundex", "nysiis", "metaphone" or
        "match rating".
    concat: bool, optional
        Remove whitespace before phonetic encoding.
    encoding: string, optional
        If bytes are given, this encoding is used to decode. Default is
        'utf-8'.
    decode_error: {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte Series is given that contains
        characters not of the given `encoding`. By default, it is 'strict',
        meaning that a UnicodeDecodeError will be raised. Other values are
        'ignore' and 'replace'.:type method: str

    Returns
    -------
    pandas.Series
        A Series with phonetic encoded values.

    Note
    ----
    The 'soundex' and 'nysiis' algorithms use the package 'jellyfish'. It can
    be installed with pip (``pip install jellyfish``).

    """

    # encoding
    if sys.version_info[0] == 2:
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if type(x) == bytes else x)

    if concat:
        s = s.str.replace('[\-\_\s]', '')

    if method == 'soundex':
        phonetic_callback = jellyfish.soundex
    elif method == 'nysiis':
        phonetic_callback = jellyfish.nysiis
    elif method == 'metaphone':
        phonetic_callback = jellyfish.metaphone
    elif method in ['match_rating', 'match rating', 'matchrating', 'match_rating_codex', 'matchratingcodex']:
        phonetic_callback = jellyfish.match_rating_codex
    else:
        raise ValueError("The algorithm '{}' is not known.".format(method))

    return s.str.upper().apply(
        lambda x: phonetic_callback(x) if pandas.notnull(x) else np.nan
    )
