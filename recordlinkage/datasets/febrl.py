from pathlib import Path

import numpy
import pandas


def _febrl_load_data(filename):
    # Internal function for loading febrl data

    filepath = Path(Path(__file__).parent, "febrl", filename)

    febrl_data = pandas.read_csv(
        filepath,
        index_col="rec_id",
        sep=",",
        engine="c",
        skipinitialspace=True,
        encoding="utf-8",
        dtype={
            "street_number": object,
            "date_of_birth": object,
            "soc_sec_id": object,
            "postcode": object,
        },
    )

    return febrl_data


def _febrl_links(df):
    """Get the links of a FEBRL dataset."""

    index = df.index.to_series()
    keys = index.str.extract(r"rec-(\d+)", expand=True)[0]

    index_int = numpy.arange(len(df))

    df_helper = pandas.DataFrame({"key": keys, "index": index_int})

    # merge the two frame and make MultiIndex.
    pairs_df = df_helper.merge(df_helper, on="key")[["index_x", "index_y"]]
    pairs_df = pairs_df[pairs_df["index_x"] > pairs_df["index_y"]]

    return pandas.MultiIndex(
        levels=[df.index.values, df.index.values],
        codes=[pairs_df["index_x"].values, pairs_df["index_y"].values],
        names=[None, None],
        verify_integrity=False,
    )


def load_febrl1(return_links=False):
    """Load the FEBRL 1 dataset.

    The Freely Extensible Biomedical Record Linkage (Febrl) package is
    distributed with a dataset generator and four datasets generated
    with the generator. This function returns the first Febrl dataset
    as a :class:`pandas.DataFrame`.

            *"This data set contains 1000 records (500 original and
            500 duplicates, with exactly one duplicate per original
            record."*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    pandas.DataFrame
        A :class:`pandas.DataFrame` with Febrl dataset1.csv. When
        return_links is True, the function returns also the true
        links. The true links are all links in the lower triangular
        part of the matrix.

    """

    df = _febrl_load_data("dataset1.csv")

    if return_links:
        links = _febrl_links(df)
        return df, links
    else:
        return df


def load_febrl2(return_links=False):
    """Load the FEBRL 2 dataset.

    The Freely Extensible Biomedical Record Linkage (Febrl) package is
    distributed with a dataset generator and four datasets generated
    with the generator. This function returns the second Febrl dataset
    as a :class:`pandas.DataFrame`.

            *"This data set contains 5000 records (4000 originals and
            1000 duplicates), with a maximum of 5 duplicates based on
            one original record (and a poisson distribution of
            duplicate records). Distribution of duplicates:
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
        A :class:`pandas.DataFrame` with Febrl dataset2.csv. When
        return_links is True, the function returns also the true
        links. The true links are all links in the lower triangular
        part of the matrix.

    """

    df = _febrl_load_data("dataset2.csv")

    if return_links:
        links = _febrl_links(df)
        return df, links
    else:
        return df


def load_febrl3(return_links=False):
    """Load the FEBRL 3 dataset.

    The Freely Extensible Biomedical Record Linkage (Febrl) package is
    distributed with a dataset generator and four datasets generated
    with the generator. This function returns the third Febrl dataset
    as a :class:`pandas.DataFrame`.

            *"This data set contains 5000 records (2000 originals and
            3000 duplicates), with a maximum of 5 duplicates based on
            one original record (and a Zipf distribution of duplicate
            records). Distribution of duplicates:
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
        A :class:`pandas.DataFrame` with Febrl dataset3.csv. When
        return_links is True, the function returns also the true
        links. The true links are all links in the lower triangular
        part of the matrix.

    """

    df = _febrl_load_data("dataset3.csv")

    if return_links:
        links = _febrl_links(df)
        return df, links
    else:
        return df


def load_febrl4(return_links=False):
    """Load the FEBRL 4 datasets.

    The Freely Extensible Biomedical Record Linkage (Febrl) package is
    distributed with a dataset generator and four datasets generated
    with the generator. This function returns the fourth Febrl dataset
    as a :class:`pandas.DataFrame`.

            *"Generated as one data set with 10000 records (5000
            originals and 5000  duplicates, with one duplicate per
            original), the originals have been split from the
            duplicates, into dataset4a.csv (containing the 5000
            original records) and dataset4b.csv (containing the
            5000 duplicate records) These two data sets can be
            used for testing linkage procedures."*

    Parameters
    ----------
    return_links: bool
        When True, the function returns also the true links.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A :class:`pandas.DataFrame` with Febrl dataset4a.csv and a pandas
        dataframe with Febrl dataset4b.csv. When return_links is True,
        the function returns also the true links.

    """

    df_a = _febrl_load_data("dataset4a.csv")
    df_b = _febrl_load_data("dataset4b.csv")

    if return_links:
        links = pandas.MultiIndex.from_arrays(
            [
                [f"rec-{i}-org" for i in range(0, 5000)],
                [f"rec-{i}-dup-0" for i in range(0, 5000)],
            ]
        )
        return df_a, df_b, links
    else:
        return df_a, df_b
