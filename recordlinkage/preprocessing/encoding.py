import sys

import jellyfish
import numpy as np
import pandas

_phonetic_algorithms = [
    {"name": "Soundex", "callback": jellyfish.soundex, "argument_names": ["soundex"]},
    {
        "name": "NYSIIS",
        "callback": jellyfish.nysiis,
        "argument_names": ["nysiis", "nyssis"],
    },
    {
        "name": "Metaphone",
        "callback": jellyfish.metaphone,
        "argument_names": ["metaphone"],
    },
    {
        "name": "Match Rating",
        "callback": jellyfish.match_rating_codex,
        "argument_names": [
            "match_rating",
            "match rating",
            "matchrating",
            "match_rating_codex",
            "matchratingcodex",
        ],
    },
]


def _list_phonetic_algorithms():
    """Return list of available phonetic algorithms."""

    return [alg["argument_names"][0] for alg in _phonetic_algorithms]


def phonetic(s, method, concat=True, encoding="utf-8", decode_error="strict"):
    """Convert names or strings into phonetic codes.

    The implemented algorithms are `soundex
    <https://en.wikipedia.org/wiki/Soundex>`_, `nysiis
    <https://en.wikipedia.org/wiki/New_York_State_Identification_and_
    Intelligence_System>`_, `metaphone
    <https://en.wikipedia.org/wiki/Metaphone>`_ or  `match_rating
    <https://en.wikipedia.org/wiki/Match_rating_approach>`_.

    Parameters
    ----------
    s : pandas.Series
        A pandas.Series with string values (often names) to encode.
    method: str
        The algorithm that is used to phonetically encode the values.
        The possible options are "soundex", "nysiis", "metaphone" or
        "match_rating".
    concat: bool, optional
        Remove whitespace before phonetic encoding.
    encoding: str, optional
        If bytes are given, this encoding is used to decode. Default
        is 'utf-8'.
    decode_error: {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte Series is given that
        contains characters not of the given `encoding`. By default,
        it is 'strict', meaning that a UnicodeDecodeError will be
        raised. Other values are 'ignore' and 'replace'.

    Returns
    -------
    pandas.Series
        A Series with phonetic encoded values.

    """

    # encoding
    if sys.version_info[0] == 2:
        s = s.apply(
            lambda x: x.decode(encoding, decode_error) if type(x) == bytes else x
        )

    if concat:
        s = s.str.replace(r"[\-\_\s]", "", regex=True)

    for alg in _phonetic_algorithms:
        if method in alg["argument_names"]:
            phonetic_callback = alg["callback"]
            break
    else:
        raise ValueError(f"The algorithm '{method}' is not known.")

    return s.str.upper().apply(
        lambda x: phonetic_callback(x) if pandas.notnull(x) else np.nan
    )
