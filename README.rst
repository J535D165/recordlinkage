Python Record Linkage Toolkit
=============================

The **Python Record Linkage Toolkit** is a library to link records in or
between data sources. The toolkit provides most of the tools needed for 
record linkage and deduplication. The package contains indexing methods, 
functions to compare records and classifiers. The package is developed 
for research and the linking of small or medium sized files.

This project is inspired by the `Freely Extensible Biomedical Record
Linkage (FEBRL) <https://sourceforge.net/projects/febrl/>`__ project,
which is a great project. In contrast with FEBRL, the recordlinkage project uses
`pandas <http://pandas.pydata.org/>`__ and `numpy <http://www.numpy.org/>`__ for 
data handling and computations. The use of *pandas*, a flexible and powerful data analysis and
manipulation library for Python, makes the record linkage process much easier
and faster. The extensive *pandas* library can be used to integrate your
record linkage directly into existing data manipulation projects.

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
smart set of candidate links with one of the built-in indexing techniques like
**blocking**. In this example, only pairs or records that agree on the surname
are included.

.. code:: python

    index = recordlinkage.Pairs(df_a, df_b)
    candidate_links = index.block('surname')

For each candidate link, compare the records with one of the
comparison or similarity algorithms in the Compare class.

.. code:: python

    c = recordlinkage.Compare(candidate_links, df_a, df_b)

    c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
    c.exact('sex', 'gender')
    c.exact('dob', 'date_of_birth')
    c.string('streetname', 'streetname', method='damerau_levenshtein', threshold=0.7)
    c.exact('place', 'placename')
    c.exact('haircolor', 'haircolor', missing_value=9)

    # The comparison vectors
    c.vectors

This Python Record Linkage Toolkit contains multiple classification alogirthms.
Plenty of the algorithms need trainings data (supervised learning) while
others are unsupervised. An example of supervised learning:

.. code:: python

    logrg = recordlinkage.LogisticRegressionClassifier()
    logrg.learn(TRAINING_COMPARISON_VECTORS, TRAINING_CLASSES)

    logrg.predict(c.vectors)

and an example of unsupervised learning (the well known ECM-algorithm):

.. code:: python

    ecm = recordlinkage.BernoulliEMClassifier()
    ecm.learn(c.vectors)

Main Features
-------------

The main features of the **recordlinkage** package are:


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
`recordlinkage.readthedocs.org <http://recordlinkage.readthedocs.org/en/latest/>`__. The documentation provides some basic usage examples like `deduplication <http://recordlinkage.readthedocs.io/en/latest/notebooks/data_deduplication.html>`__ and `linking <http://recordlinkage.readthedocs.io/en/latest/notebooks/link_two_dataframes.html>`__ census data. More examples are coming soon. If you do have interesting examples to share, let us know.


Dependencies, installation and license
--------------------------------------

|pypi| |travis| |codecov|

.. |travis| image:: https://travis-ci.org/J535D165/recordlinkage.svg?branch=master
    :target: https://travis-ci.org/J535D165/recordlinkage

.. |pypi| image:: https://badge.fury.io/py/recordlinkage.svg
    :target: https://pypi.python.org/pypi/recordlinkage/
    
.. |codecov| image:: https://codecov.io/gh/J535D165/recordlinkage/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/J535D165/recordlinkage

The following packages are required. You probably have most of them already
installed.

-  `numpy <http://www.numpy.org>`__
-  `pandas <https://github.com/pydata/pandas>`__
-  `scipy <https://www.scipy.org/>`__
-  `sklearn <http://scikit-learn.org/>`__
-  `jellyfish <https://github.com/jamesturk/jellyfish>`__: Needed for
   approximate string comparison and string encoding. 

The following packages are recommanded

- numexpr: Used to speed up numeric comparisons. 

Install the package with pip

.. code:: sh

    pip install recordlinkage

The license for this record linkage tool is GPLv3.

Need help?
----------

Stuck on your record linkage code or problem? Any other questions? Don't hestitate to send me an email (jonathandebruinhome@gmail.com).
