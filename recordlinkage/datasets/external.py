# The function get_data_home() and clear_data_home() are based on
# SciKit-Learn https://git.io/fjT70. See the 3-clause BSD license.

import shutil
import zipfile
from io import BytesIO
from os import environ
from pathlib import Path
from urllib.request import urlopen

import pandas


def get_data_home(data_home=None):
    """Return the path of the Record Linkage data folder.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times. By default the data dir
    is set to a folder named 'rl_data' in the user
    home folder.
    Alternatively, it can be set by the 'RL_DATA' environment
    variable or programmatically by giving an explicit folder
    path. The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically
    created.

    Parameters
    ----------
    data_home : str | None
        The path to recordlinkage data folder.
    """
    if data_home is None:
        data_home = environ.get("RL_DATA", Path("~", "rl_data"))
    data_home = Path(data_home).expanduser()

    if not data_home.exists():
        data_home.mkdir(parents=True, exist_ok=True)

    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str | None
        The path to recordlinkage data folder.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(str(data_home))


def load_krebsregister(block=None, missing_values=None, shuffle=True):
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
        blocks are the blocks explained in the description. Default
        all 1 to 10.
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

    if block is None:
        block = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # If the data is not found, download it.
    for i in range(1, 11):
        filepath = Path(get_data_home(), "krebsregister", f"block_{i}.zip")

        if not filepath.is_file():
            _download_krebsregister()
            break

    if isinstance(block, (list, tuple)):
        data = pandas.concat([_krebsregister_block(bl) for bl in block])
    else:
        data = _krebsregister_block(block)

    if shuffle:
        data = data.sample(frac=1, random_state=535)

    match_index = data.index[data["is_match"]]
    del data["is_match"]

    if pandas.notnull(missing_values):
        data.fillna(missing_values, inplace=True)

    return data, match_index


def _download_krebsregister():
    zip_file_url = (
        "http://archive.ics.uci.edu/ml/" "machine-learning-databases/00210/donation.zip"
    )

    folder = Path(get_data_home(), "krebsregister")

    try:
        print(f"Downloading data to {folder}.")
        r = urlopen(zip_file_url).read()

        # unzip the content and put it in the krebsregister folder
        z = zipfile.ZipFile(BytesIO(r))
        z.extractall(str(folder))

        print("Data download succesfull.")

    except Exception as e:
        print("Issue with downloading the data:", e)


def _krebsregister_block(block):
    if block not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError(
            "Argument 'block' has to be integer in "
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] or list of integers."
        )

    fp_i = Path(get_data_home(), "krebsregister", f"block_{block}.zip")

    data_block = pandas.read_csv(
        fp_i, index_col=["id_1", "id_2"], na_values="?", compression="zip"
    )

    data_block.columns = [
        "cmp_firstname1",
        "cmp_firstname2",
        "cmp_lastname1",
        "cmp_lastname2",
        "cmp_sex",
        "cmp_birthday",
        "cmp_birthmonth",
        "cmp_birthyear",
        "cmp_zipcode",
        "is_match",
    ]
    data_block.index.names = ["id1", "id2"]

    return data_block
