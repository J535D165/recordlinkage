import os
import zipfile

import pandas

from six import BytesIO
from six.moves.urllib.request import urlopen


def load_krebsregister(block=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       missing_values=None, shuffle=True):
    """Load the Krebsregister dataset.

    This dataset of comparison patterns was obtained in a
    epidemiological cancer study in Germany. The comparison patterns
    were created by the Institute for Medical Biostatistics,
    Epidemiology and Informatics (IMBEI) and the University Medical
    Center of Johannes Gutenberg University (Mainz, Germany). The
    dataset is available for research online.

    "The records represent individual data including first and
    family name, sex, date of birth and postal code, which were
    collected through iterative insertions in the course of
    several years. The comparison patterns in this data set are
    based on a sample of 100.000 records dating from 2005 to 2008.
    Data pairs were classified as "match" or "non-match" during
    an extensive manual review where several documentarists were
    involved.  The resulting classification formed the basis for
    assessing the quality of the registry's own record linkage
    procedure.

    In order to limit the amount of patterns a blocking procedure
    was applied, which selects only record pairs that meet
    specific agreement conditions. The results of the following
    six blocking iterations were merged together:

    - Phonetic equality of first name and family name, equality of
      date of birth.
    - Phonetic equality of first name, equality of day of birth.
    - Phonetic equality of first name, equality of month of birth.
    - Phonetic equality of first name, equality of year of birth.
    - Equality of complete date of birth.
    - Phonetic equality of family name, equality of sex.

    This procedure resulted in 5.749.132 record pairs, of which
    20.931 are matches. The data set is split into 10 blocks of
    (approximately) equal size and ratio of matches to
    non-matches."

    Parameters
    ----------
    block : int, list
        An integer or a list with integers between 1 and 10. The
        blocks are the blocks explained in the description.
    missing_values : object, int, float
        The value of the missing values. Default NaN.
    shuffle : bool
        Shuffle the record pairs. Default True.

    Returns
    -------
    (pandas.DataFrame, pandas.MultiIndex)
        A pandas.DataFrame with comparison vectors and a
        pandas.MultiIndex with the indices of the matches.

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

    if shuffle:
        data = data.sample(frac=1, random_state=535)

    match_index = data.index[data['is_match']]
    del data['is_match']

    if pandas.notnull(missing_values):
        data.fillna(missing_values, inplace=True)

    return data, match_index


def _download_krebsregister():

    zip_file_url = "http://archive.ics.uci.edu/ml/" \
        "machine-learning-databases/00210/donation.zip"

    try:
        print("Start downloading the data.")
        r = urlopen(zip_file_url).read()

        # unzip the content and put it in the krebsregister folder
        z = zipfile.ZipFile(BytesIO(r))
        z.extractall(os.path.join(os.path.dirname(__file__), 'krebsregister'))

        print("Data download succesfull.")

    except Exception as e:
        print("Issue with downloading the data:", e)


def _krebsregister_block(block):

    if block not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError(
            "Argument 'block' has to be integer in "
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] or list of integers.")

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
