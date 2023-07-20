<div align="center">
  <img src="https://raw.githubusercontent.com/J535D165/recordlinkage/master/docs/images/recordlinkage-banner-transparent.svg"><br>
</div>

# RecordLinkage: powerful and modular Python record linkage toolkit

[![Pypi Version](https://badge.fury.io/py/recordlinkage.svg)](https://pypi.python.org/pypi/recordlinkage/)
[![Github Actions CI Status](https://github.com/J535D165/recordlinkage/workflows/tests/badge.svg?branch=master)](https://github.com/J535D165/recordlinkage/actions)
[![Code Coverage](https://codecov.io/gh/J535D165/recordlinkage/branch/master/graph/badge.svg)](https://codecov.io/gh/J535D165/recordlinkage)
[![Documentation Status](https://readthedocs.org/projects/recordlinkage/badge/?version=latest)](https://recordlinkage.readthedocs.io/en/latest/?badge=latest)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3559042.svg)](https://doi.org/10.5281/zenodo.3559042)

**RecordLinkage** is a powerful and modular record linkage toolkit to
link records in or between data sources. The toolkit provides most of
the tools needed for record linkage and deduplication. The package
contains indexing methods, functions to compare records and classifiers.
The package is developed for research and the linking of small or medium
sized files.

This project is inspired by the [Freely Extensible Biomedical Record
Linkage (FEBRL)](https://sourceforge.net/projects/febrl/) project, which
is a great project. In contrast with FEBRL, the recordlinkage project
uses [pandas](http://pandas.pydata.org/) and
[numpy](http://www.numpy.org/) for data handling and computations. The
use of *pandas*, a flexible and powerful data analysis and manipulation
library for Python, makes the record linkage process much easier and
faster. The extensive *pandas* library can be used to integrate your
record linkage directly into existing data manipulation projects.

One of the aims of this project is to make an easily extensible record
linkage framework. It is easy to include your own indexing algorithms,
comparison/similarity measures and classifiers.

## Basic linking example

Import the `recordlinkage` module with all important tools for record
linkage and import the data manipulation framework **pandas**.

``` python
import recordlinkage
import pandas
```

Load your data into pandas DataFrames.

``` python
df_a = pandas.DataFrame(YOUR_FIRST_DATASET)
df_b = pandas.DataFrame(YOUR_SECOND_DATASET)
```

Comparing all record can be computationally intensive. Therefore, we
make set of candidate links with one of the built-in indexing techniques
like **blocking**. In this example, only pairs of records that agree on
the surname are returned.

``` python
indexer = recordlinkage.Index()
indexer.block('surname')
candidate_links = indexer.index(df_a, df_b)
```

For each candidate link, compare the records with one of the comparison
or similarity algorithms in the Compare class.

``` python
c = recordlinkage.Compare()

c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
c.exact('sex', 'gender')
c.date('dob', 'date_of_birth')
c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
c.exact('place', 'placename')
c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

# The comparison vectors
feature_vectors = c.compute(candidate_links, df_a, df_b)
```

Classify the candidate links into matching or distinct pairs based on
their comparison result with one of the [classification
algorithms](https://recordlinkage.readthedocs.io/en/latest/ref-classifiers.html).
The following code classifies candidate pairs with a Logistic Regression
classifier. This (supervised machine learning) algorithm requires
training data.

``` python
logrg = recordlinkage.LogisticRegressionClassifier()
logrg.fit(TRAINING_COMPARISON_VECTORS, TRAINING_PAIRS)

logrg.predict(feature_vectors)
```

The following code shows the classification of candidate pairs with the
Expectation-Conditional Maximisation (ECM) algorithm. This variant of
the Expectation-Maximisation algorithm doesn't require training data
(unsupervised machine learning).

``` python
ecm = recordlinkage.ECMClassifier()
ecm.fit_predict(feature_vectors)
```

## Main Features

The main features of this Python record linkage toolkit are:

-   Clean and standardise data with easy to use tools
-   Make pairs of records with smart indexing methods such as
    **blocking** and **sorted neighbourhood indexing**
-   Compare records with a large number of comparison and similarity
    measures for different types of variables such as strings, numbers
    and dates.
-   Several classifications algorithms, both supervised and unsupervised
    algorithms.
-   Common record linkage evaluation tools
-   Several built-in datasets.

## Documentation

The most recent documentation and API reference can be found at
[recordlinkage.readthedocs.org](http://recordlinkage.readthedocs.org/en/latest/).
The documentation provides some basic usage examples like
[deduplication](http://recordlinkage.readthedocs.io/en/latest/guides/data_deduplication.html)
and
[linking](http://recordlinkage.readthedocs.io/en/latest/guides/link_two_dataframes.html)
census data. More examples are coming soon. If you do have interesting
examples to share, let us know.

## Installation

The Python Record linkage Toolkit requires Python 3.8 or higher. Install the
package easily with pip

``` sh
pip install recordlinkage
```

The toolkit depends on popular packages like
[Pandas](https://github.com/pydata/pandas),
[Numpy](http://www.numpy.org), [Scipy](https://www.scipy.org/) and,
[Scikit-learn](http://scikit-learn.org/). A complete list of
dependencies can be found in the [installation
manual](https://recordlinkage.readthedocs.io/en/latest/installation.html)
as well as recommended and optional dependencies.

## License

The license for this record linkage tool is BSD-3-Clause.

## Citation

Please cite this package when being used in an academic context. Ensure
that the DOI and version match the installed version. Citatation styles
can be found on the publishers website
[10.5281/zenodo.3559042](https://doi.org/10.5281/zenodo.3559042).

``` text
@software{de_bruin_j_2019_3559043,
  author       = {De Bruin, J},
  title        = {{Python Record Linkage Toolkit: A toolkit for
                   record linkage and duplicate detection in Python}},
  month        = dec,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0.14},
  doi          = {10.5281/zenodo.3559043},
  url          = {https://doi.org/10.5281/zenodo.3559043}
}
```

## Need help?

Stuck on your record linkage code or problem? Any other questions? Don't
hestitate to send me an email (<jonathandebruinos@gmail.com>).
