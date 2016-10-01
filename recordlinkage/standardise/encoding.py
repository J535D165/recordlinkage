from __future__ import division
# from __future__ import absolute_import
# from __future__ import unicode_literals

import warnings
import os
import sys

import numpy as np
import pandas

import itertools

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


def gender(
    names, years=None, method="ssa", countries=None,
    labels=("male", "female"), q=0.1
):
    """

    Predict the gender based on the first name. This tool is based on
    https://github.com/ropensci/gender and uses their data. The prediction of
    the sex for each name uses `Bayes' theorem
    <https://en.wikipedia.org/wiki/Bayes%27_theorem>`_.

    :param names: The given names.
    :param years: The birth year of the first names whose gender is to be
            predicted. This argument can be either a single year, a range of
            years in the form [1880, 1900]. If no value is specified, then for
            the ``ssa`` method it will use the period 1932 to 2012.
    :param method: This value determines the data set that is used to predict
            the gender of the name. The ``ssa`` method looks up names based
            from the U.S. Social Security Administration baby name data.
    :param countries: The country.
    :param labels: The label given for male and female prediction. Default:
            ("male", "female").
    :param sex_ratio: The ratio of males to females (`sex ratio
            <https://en.wikipedia.org/wiki/Sex_ratio>`_) in the newborn
            population. Default 1.07
    :param q: The maximum error probability for a name.

    :type names: pandas.Series
    :type years: str, int, (list, tuple), (pandas.Series, numpy.array)
    :type method: "ssa"
    :type countries: "United States"
    :type labels: 2-tuple or list
    :type q: float
    :type sex_ratio: float

    :return: The gender predictions for each row.
    :rtype: pandas.Series

    Example:

    Predict the sex of the records in the first FEBRL dataset:

    .. code:: python

        >>> from recordlinkage.datasets import load_febrl1
        >>> from recordlinkage.standardise import gender

        >>> data = load_febrl1()
        >>> data["sex"] = gender(data['given_name'], years=[1990, 2010], method="ssa")

        >>> print (data["sex"].head())

        rec_id
        rec-223-org        NaN
        rec-122-org       male
        rec-373-org       male
        rec-10-dup-0    female
        rec-227-org       male
        Name: sex, dtype: object

    """

    if method == "ssa" and (countries == "United States" or countries is None):
        return gender_ssa(names, years, labels=labels, q=q)
    else:
        raise NotImplementedError(
            "This chosen method {} is not known.".format(method))


def gender_ssa(
    names, years=None, labels=("male", "female"), q=0.49, sex_ratio=1.07
):

    filepath = os.path.join(os.path.dirname(__file__),
                            'data', 'ssa_national.zip')
    fileurl = "https://github.com/J535D165/recordlinkage/raw/master/recordlinkage/standardise/data/ssa_national.zip"

    if os.path.exists(filepath):
        ssa_national = pandas.read_csv(filepath, compression="zip")
    else:
        try:
            ssa_national = pandas.read_csv(fileurl, compression="zip")
        except Exception:
            raise Exception("Unable the load the dataset.")

    ssa_min_year = int(ssa_national["year"].min())
    ssa_max_year = int(ssa_national["year"].max())

    if isinstance(years, (pandas.Series, np.ndarray)):  # pandas.Series

        if len(names) != len(years):
            raise ValueError("Length of names and years are not the same.")

        years = [int(years.min()), int(years.max())]

        if (years[1] > ssa_max_year) or (years[0] < ssa_min_year):
            warnings.warn(
                "years outside the range {} - {}. " +
                "Using truncated average.".format(
                    ssa_min_year, ssa_max_year))
        else:
            s_years = pandas.Series(years)

    elif isinstance(years, (list, tuple)):
        years = [min(years), max(years)]

    if years is not None:  # None or empty list
        years = [1930, ssa_max_year]

    elif not isinstance(years, (list, tuple)):
        years = [years, years]

    else:
        raise ValueError("years is not a valid type.")

    if (years[1] > ssa_max_year) or (years[0] < ssa_min_year):
        raise ValueError(
            "Use years in the range {} - {}".format(ssa_min_year, ssa_max_year)
        )

    # Make subset of relevant years
    ssa_national_subset = ssa_national[
        (ssa_national["year"] >= years[0]) &
        (ssa_national["year"] <= years[1])
    ]

    # Correct the skew. Different than in https://github.com/ropensci/gender
    if 's_years' not in locals():
        ssa_national_grouped = ssa_national_subset.groupby(
            'name')[['female', 'male']].sum()
    else:
        ssa_national_grouped = ssa_national_subset.copy().set_index([
            'name', 'year'])

    # compute sex ratio https://en.wikipedia.org/wiki/Sex_ratio with
    if not sex_ratio:
        P_male = ssa_national_grouped['male'] \
            / (ssa_national_grouped['male'] + ssa_national_grouped['female'])
        P_female = ssa_national_grouped['female'] \
            / (ssa_national_grouped['male'] + ssa_national_grouped['female'])
        sex_ratio = P_male / P_female

    P_name_given_male = ssa_national_grouped[
        'male'] / ssa_national_grouped['male'].sum()
    P_name_given_female = ssa_national_grouped[
        'female'] / ssa_national_grouped['female'].sum()

    P_male_given_name = (P_name_given_male * sex_ratio) / \
        ((P_name_given_male * sex_ratio) + P_name_given_female)

    # Proportion
    if q <= 0 or q >= 0.5:
        raise ValueError("q must be a value between 0 and 0.5. ")

    ssa_national_grouped['prediction'] = np.nan
    ssa_national_grouped.loc[P_male_given_name >=
                             1 - q, "prediction"] = labels[0]
    ssa_national_grouped.loc[P_male_given_name <= q, "prediction"] = labels[1]

    #
    if 's_years' in locals():
        return pandas.Series(
            ssa_national_grouped.loc[pandas.MultiIndex.from_arrays(
                [names.fillna(""), s_years]), "prediction"].values,
            index=names.index.values
        )
    else:
        return pandas.Series(
            ssa_national_grouped.loc[names.fillna(""), "prediction"].values,
            index=names.index
        )


def similar_values(s, threshold=0.8):
    """
    similar_values(threshold=0.8)

    Group strings with high similarities.

    :param threshold: Two strings with similarity above this threshold are
            considered to be the same string. The threshold is a value equal
            or between 0 and 1. Default 0.8.
    :param inplace: If True, replace the current strings by their cleaned
            variant. Default: True.

    :return: A Series of strings.
    :rtype: pandas.Series

    """
    try:
        import jellyfish
    except ImportError:
        print ("Install jellyfish to use string encoding.")

    replace_tuples = []

    for pair in itertools.combinations(
        self[self.notnull()].astype(unicode).unique(),
        2
    ):

        sim = 1 - \
            jellyfish.levenshtein_distance(
                pair[0], pair[1]) / np.max([len(pair[0]), len(pair[1])])

        if (sim >= threshold):
            replace_tuples.append(pair)

    # This is not a very clever solution I think. Don't known how to solve it
    # atm: connected_components?
    for pair in replace_tuples:

        if (sum(self == pair[0]) > sum(self == pair[1])):
            self = pandas.Series(self.str.replace(pair[1], pair[0]))
        else:
            self = pandas.Series(self.str.replace(pair[0], pair[1]))

    return pandas.Series(string)
