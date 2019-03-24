Python Record Linkage Toolkit
=============================

|pypi| |travis| |codecov| |docs|

.. |travis| image:: https://travis-ci.org/J535D165/recordlinkage.svg?branch=master
  :target: https://travis-ci.org/J535D165/recordlinkage
  :alt: TravisCI Status

.. |pypi| image:: https://badge.fury.io/py/recordlinkage.svg
  :target: https://pypi.python.org/pypi/recordlinkage/
  :alt: Pypi Version
    
.. |codecov| image:: https://codecov.io/gh/J535D165/recordlinkage/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/J535D165/recordlinkage
  :alt: Code Coverage

.. |docs| image:: https://readthedocs.org/projects/recordlinkage/badge/?version=latest
  :target: https://recordlinkage.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

The **Python Record Linkage Toolkit** is a library to link records in or
between data sources. The toolkit provides most of the tools needed for
record linkage and deduplication. The package contains indexing methods,
functions to compare records and classifiers. The package is developed for
research and the linking of small or medium sized files.

This project is inspired by the `Freely Extensible Biomedical Record Linkage
(FEBRL) <https://sourceforge.net/projects/febrl/>`__ project, which is a great
project. In contrast with FEBRL, the recordlinkage project uses `pandas
<http://pandas.pydata.org/>`__ and `numpy <http://www.numpy.org/>`__ for data
handling and computations. The use of *pandas*, a flexible and powerful data
analysis and manipulation library for Python, makes the record linkage process
much easier and faster. The extensive *pandas* library can be used to
integrate your record linkage directly into existing data manipulation
projects.

One of the aims of this project is to make an easily extensible record 
linkage framework. It is easy to include your own indexing algorithms,
comparison/similarity measures and classifiers.


Basic linking example
---------------------

Import the ``recordlinkage`` module with all important tools for record
linkage and import the data manipulation framework **pandas**.

.. code:: python

    import recordlinkage
    import pandas

Load your data into pandas DataFrames. 

.. code:: python

    df_a = pandas.DataFrame(YOUR_FIRST_DATASET)
    df_b = pandas.DataFrame(YOUR_SECOND_DATASET)

Comparing all record can be computationally intensive. Therefore, we make 
set of candidate links with one of the built-in indexing techniques like
**blocking**. In this example, only pairs of records that agree on the surname
are returned.

.. code:: python

    indexer = recordlinkage.Index()
    indexer.block('surname')
    candidate_links = indexer.index(df_a, df_b)

For each candidate link, compare the records with one of the
comparison or similarity algorithms in the Compare class.

.. code:: python

    c = recordlinkage.Compare()

    c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
    c.exact('sex', 'gender')
    c.date('dob', 'date_of_birth')
    c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
    c.exact('place', 'placename')
    c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

    # The comparison vectors
    feature_vectors = c.compute(candidate_links, df_a, df_b)

This Python Record Linkage Toolkit contains multiple classification algorithms.
Plenty of the algorithms do need training data (supervised learning) while
others are unsupervised. An example of supervised learning:

.. code:: python

    logrg = recordlinkage.LogisticRegressionClassifier()
    logrg.fit(TRAINING_COMPARISON_VECTORS, TRAINING_CLASSES)

    logrg.predict(feature_vectors)

and an example of unsupervised learning (the well known ECM-algorithm):

.. code:: python

    ecm = recordlinkage.ECMClassifier()
    ecm.fit_predict(feature_vectors)

Main Features
-------------

The main features of the **Python Record Linkage Toolkit** are:


-  Clean and standardise data with easy to use tools
-  Make pairs of records with smart indexing methods such as
   **blocking** and **sorted neighbourhood indexing**
-  Compare records with a large number of comparison and similarity
   measures for different types of variables such as strings, numbers and dates.
-  Several classifications algorithms, both supervised and unsupervised
   algorithms.
-  Common record linkage evaluation tools
-  Several built-in datasets. 

Documentation 
-------------

The most recent documentation and API reference can be found at
`recordlinkage.readthedocs.org
<http://recordlinkage.readthedocs.org/en/latest/>`__. The documentation
provides some basic usage examples like deduplication_ and linking_ census
data. More examples are coming soon. If you do have interesting examples to
share, let us know.

.. _deduplication: http://recordlinkage.readthedocs.io/en/latest/notebooks/data_deduplication.html
.. _linking: http://recordlinkage.readthedocs.io/en/latest/notebooks/link_two_dataframes.html

Dependencies, installation and license
--------------------------------------

Install the Python Record Linkage Toolkit easily with pip

.. code:: sh

    pip install recordlinkage

The toolkit depends on Pandas_ (>=18.0), Numpy_, `Scikit-learn`_, Scipy_ and
Jellyfish_. You probably have most of them already installed. The package
``jellyfish`` is used for approximate string comparing and string encoding.
The package Numexpr_ is an optional dependency to speed up numerical
comparisons.

.. _Numpy: http://www.numpy.org
.. _Pandas: https://github.com/pydata/pandas
.. _Scipy: https://www.scipy.org/
.. _Scikit-learn: http://scikit-learn.org/
.. _Jellyfish: https://github.com/jamesturk/jellyfish
.. _Numexpr: https://github.com/pydata/numexpr

The license for this record linkage tool is BSD-3-Clause.

Need help?
----------

Stuck on your record linkage code or problem? Any other questions? Don't
hestitate to send me an email (jonathandebruinos@gmail.com).
