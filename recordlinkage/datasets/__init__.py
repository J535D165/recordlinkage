
import pandas

import os
import zipfile
from six import BytesIO


def load_krebsregister(block=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """

    This dataset of comparison patterns was obtained in a epidemiological
    cancer study in Germany. The comparison patterns were created by the
    Institute for Medical Biostatistics, Epidemiology and Informatics (IMBEI)
    and the University Medical Center of Johannes Gutenberg University
    (Mainz, Germany). The dataset is available for research online.

            *"The records represent individual data including first and family
            name, sex, date of birth and postal code, which were collected
            through iterative insertions in the course of several years. The
            comparison patterns in this data set are based on a sample of
            100.000 records dating from 2005 to 2008. Data pairs were
            classified as "match" or "non-match" during  an extensive manual
            review where several documentarists were involved.  The resulting
            classification formed the basis for assessing the quality of the
            registry's own record linkage procedure.*

            *In order to limit the amount of patterns a blocking procedure was
            applied, which selects only record pairs that meet specific
            agreement conditions. The results of the following six blocking
            iterations were merged together:*

            1. *Phonetic equality of first name and family name, equality of
                    date of birth.*
            2. *Phonetic equality of first name, equality of day of birth.*
            3. *Phonetic equality of first name, equality of month of birth.*
            4. *Phonetic equality of first name, equality of year of birth.*
            5. *Equality of complete date of birth.*
            6. *Phonetic equality of family name, equality of sex.*

            *This procedure resulted in 5.749.132 record pairs, of which
            20.931 are matches. The data set is split into 10 blocks of
            (approximately) equal size and ratio of matches to non-matches."*

    :param block: An integer or a list with integers between 1 and 10. The
            blocks are the blocks explained in the description.

    :return: A data frame with comparison vectors and a multi index with the
            indices of the matches.
    :rtype: (pandas.DataFrame, pandas.MultiIndex)

    """

    # If the data is not found, download it.
    for i in range(1, 11):

        filepath = os.path.join(os.path.dirname(__file__),
                                'krebsregister', 'block_{}.zip'.format(i))

        if not os.path.exists(filepath):
            _download_krebsregister()
            break

    if isinstance(block, (list, tuple)):

        data = pandas.concat([_krebsregister_block(bl) for bl in block])
    else:

        data = _krebsregister_block(block)

    match_index = data.index[data['is_match']]
    del data['is_match']

    return data, match_index


def _download_krebsregister():

    # Try to import requests
    import requests

    zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00210/donation.zip"

    try:
        print("Start downloading the data.")
        r = requests.get(zip_file_url)

        # unzip the content and put it in the krebsregister folder
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(os.path.join(os.path.dirname(__file__), 'krebsregister'))

        print("Data download succesfull.")

    except Exception as e:
        print("Issue with downloading the data:", e)


def _krebsregister_block(block):

    if block not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError(
            "Argument 'block' has to be integer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] or list of integers.")

    fp_i = os.path.join(os.path.dirname(__file__),
                        'krebsregister', 'block_{}.zip'.format(block))

    data_block = pandas.read_csv(
        fp_i,
        index_col=['id_1', 'id_2'],
        na_values='?',
        compression='zip')

    data_block.columns = [
        'cmp_firstname1', 'cmp_firstname2', 'cmp_lastname1', 'cmp_lastname2',
        'cmp_sex', 'cmp_birthday', 'cmp_birthmonth', 'cmp_birthyear',
        'cmp_zipcode', 'is_match']
    data_block.index.names = ['id1', 'id2']

    return data_block


def load_febrl1():
    """

    The Freely Extensible Biomedical Record Linkage (Febrl) package was
    distributed with a dataset generator and four datasets generated with the
    generator. This functions returns the first Febrl dataset as a pandas
    DataFrame.

            *"This data set contains 1000 records (500 original and 500
            duplicates, with exactly one duplicate per original record."*

    :return: A pandas DataFrame with Febrl dataset1.csv.
    :rtype: pandas.DataFrame

    """

    return _load_febrl_data('dataset1.csv')


def load_febrl2():
    """

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

    :return: A pandas DataFrame with Febrl dataset2.csv.
    :rtype: pandas.DataFrame

    """

    return _load_febrl_data('dataset2.csv')


def load_febrl3():
    """

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

    :return: A pandas DataFrame with Febrl dataset3.csv.
    :rtype: pandas.DataFrame

    """

    return _load_febrl_data('dataset3.csv')


def load_febrl4():
    """

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

    :return: A pandas DataFrame with Febrl dataset4a.csv and a pandas
            DataFrame with Febrl dataset4b.csv.
    rtype: (pandas.DataFrame, pandas.DataFrame)

    """

    return _load_febrl_data('dataset4a.csv'), _load_febrl_data('dataset4b.csv')


def _load_febrl_data(filename):
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
