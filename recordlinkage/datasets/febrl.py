import os

import pandas


def _febrl_load_data(filename):
    # Internal function for loading febrl data

    filepath = os.path.join(os.path.dirname(__file__), 'febrl', filename)

    febrl_data = pandas.read_csv(filepath,
                                 index_col="rec_id",
                                 sep=",",
                                 engine='c',
                                 skipinitialspace=True,
                                 encoding='utf-8',
                                 dtype={
                                     "street_number": object,
                                     "date_of_birth": object,
                                     "soc_sec_id": object,
                                     "postcode": object
                                 })

    return febrl_data


def _febrl_links(df):
    """Get the links of a FEBRL dataset."""

    df_empty = df[[]].reset_index()
    df_empty['key'] = df_empty['rec_id'].str. \
        extract(r'rec-(\d+)', expand=True)[0]

    # split the dataframe (org and dup)
    org_bool = df_empty['rec_id'].str.endswith("org")

    # merge the two frame and make MultiIndex.
    pairs = df_empty[org_bool].merge(df_empty[~org_bool], on='key')
    pairs_mi = pairs.set_index(['rec_id_x', 'rec_id_y']).index
    pairs_mi.names = ['rec_id', 'rec_id']

    return pairs_mi


def load_febrl1(return_links=False):
    """FEBRL dataset 1

    The Freely Extensible Biomedical Record Linkage (Febrl) package was
    distributed with a dataset generator and four datasets generated with the
    generator. This functions returns the first Febrl dataset as a pandas
    DataFrame.

            *"This data set contains 1000 records (500 original and 500
            duplicates, with exactly one duplicate per original record."*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with Febrl dataset1.csv. When return_links is True,
        the function returns also the true links.

    """

    df = _febrl_load_data('dataset1.csv')

    if return_links:
        links = pandas.MultiIndex.from_arrays([
            ["rec-{}-org".format(i) for i in range(0, 500)],
            ["rec-{}-dup-0".format(i) for i in range(0, 500)]]
        )
        return df, links
    else:
        return df


def load_febrl2(return_links=False):
    """FEBRL dataset 2

    The Freely Extensible Biomedical Record Linkage (Febrl) package was
    distributed with a dataset generator and four datasets generated with the
    generator. This functions returns the second Febrl dataset as a pandas
    DataFrame.

            *"This data set contains 5000 records (4000 originals and 1000
            duplicates), with a maximum of 5 duplicates based on one original
            record (and a poisson distribution of duplicate records).
            Distribution of duplicates:
            19 originals records have 5 duplicate records
            47 originals records have 4 duplicate records
            107 originals records have 3 duplicate records
            141 originals records have 2 duplicate records
            114 originals records have 1 duplicate record
            572 originals records have no duplicate record"*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with Febrl dataset2.csv. When return_links is True,
        the function returns also the true links.

    """

    df = _febrl_load_data('dataset2.csv')

    if return_links:
        links = _febrl_links(df)
        return df, links
    else:
        return df


def load_febrl3(return_links=False):
    """FEBRL dataset 3

    The Freely Extensible Biomedical Record Linkage (Febrl) package was
    distributed with a dataset generator and four datasets generated with the
    generator. This functions returns the third Febrl dataset as a pandas
    DataFrame.

            *"This data set contains 5000 records (2000 originals and 3000
            duplicates), with a maximum of 5 duplicates based on one original
            record (and a Zipf distribution of duplicate records).
            Distribution of duplicates:
            168 originals records have 5 duplicate records
            161 originals records have 4 duplicate records
            212 originals records have 3 duplicate records
            256 originals records have 2 duplicate records
            368 originals records have 1 duplicate record
            1835 originals records have no duplicate record"*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with Febrl dataset3.csv. When return_links is True,
        the function returns also the true links.

    """

    df = _febrl_load_data('dataset3.csv')

    if return_links:
        links = _febrl_links(df)
        return df, links
    else:
        return df


def load_febrl4(return_links=False):
    """FEBRL dataset 4

    The Freely Extensible Biomedical Record Linkage (Febrl) package was
    distributed with a dataset generator and four datasets generated with the
    generator. This  functions returns the fourth Febrl dataset as a pandas
    DataFrame.

            *"Generated as one data set with 10000 records (5000 originals and
            5000  duplicates, with one duplicate per original), the originals
            have been split from the duplicates, into dataset4a.csv
            (containing the 5000 original records) and dataset4b.csv
            (containing the 5000 duplicate records) These two data sets can be
            used for testing linkage procedures."*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A pandas DataFrame with Febrl dataset4a.csv and a pandas dataframe
        with Febrl dataset4b.csv. When return_links is True,
        the function returns also the true links.

    """

    df_a = _febrl_load_data('dataset4a.csv')
    df_b = _febrl_load_data('dataset4b.csv')

    if return_links:
        links = pandas.MultiIndex.from_arrays([
            ["rec-{}-org".format(i) for i in range(0, 5000)],
            ["rec-{}-dup-0".format(i) for i in range(0, 5000)]]
        )
        return df_a, df_b, links
    else:
        return df_a, df_b
