*****
About
*****

Introduction
============

The **Python Record Linkage Toolkit** is a library to link records in or
between data sources. The toolkit provides most of the tools needed for 
record linkage and deduplication. The package contains indexing methods, 
functions to compare records and classifiers. The package is developed 
for research and the linking of small or medium sized files.

The project is inspired by the `Freely Extensible Biomedical Record Linkage
(FEBRL) <https://sourceforge.net/projects/febrl/>`__ project, which is a great
project. In contrast with FEBRL, the recordlinkage project makes extensive use
of data manipulation tools like `pandas <http://pandas.pydata.org/>`__ and
`numpy <http://www.numpy.org/>`__. The use of *pandas*, a flexible and
powerful data analysis and manipulation library for Python, makes the record
linkage process much easier and faster. The extensive *pandas* library can be
used to integrate your record linkage directly into existing data manipulation
projects.

One of the aims of this project is to make an extensible record linkage
framework. It is easy to include your own indexing algorithms,
comparison/similarity measures and classifiers. The main features of the
Python Record Linkage Toolkit are:

-  Clean and standardise data with easy to use tools
-  Make pairs of records with smart indexing methods such as
   **blocking** and **sorted neighbourhood indexing**
-  Compare records with a large number of comparison and similarity measures
   for different types of variables such as strings, numbers and dates.
-  Several classifications algorithms, both supervised and unsupervised
   algorithms.
-  Common record linkage evaluation tools
-  Several built-in datasets. 


What is record linkage?
=======================

The term record linkage is used to indicate the procedure of bringing together
information from two or more records that are believed to belong to the same
entity. Record linkage is used to link data from multiple data sources or to
find duplicates in a single data source. In computer science, record linkage
is also known as data matching or deduplication (in case of search duplicate
records within a single file).

In record linkage, the attributes of the entity (stored in a record) are used
to link two or more records. Attributes can be unique entity identifiers (SSN,
license plate number), but also attributes like (sur)name, date of birth and
car model/colour. The record linkage procedure can be represented as a
workflow [Christen, 2012]. The steps are: cleaning, indexing, comparing,
classifying and evaluation. If needed, the classified record pairs flow back
to improve the previous step. The Python Record Linkage Toolkit follows this
workflow.

.. seealso::

    *Christen, Peter. 2012. Data matching: concepts and techniques for record 
    linkage, entity resolution, and duplicate detection. Springer Science & 
    Business Media.*

    *Fellegi, Ivan P and Alan B Sunter. 1969. “A theory for record linkage.” 
    Journal of the American Statistical Association 64(328):1183–1210.*

    *Dunn, Halbert L. 1946. “Record linkage.” American Journal of Public 
    Health and the Nations Health 36(12):1412–1416.*

    *Herzog, Thomas N, Fritz J Scheuren and William E Winkler. 2007. Data 
    quality and record linkage techniques. Vol. 1 Springer.*

How to link records?
====================

Import the ``recordlinkage`` module with all important tools for record
linkage and import the data manipulation framework **pandas**.

.. code:: python

    import recordlinkage
    import pandas

Consider that you try to link two datasets with personal information
like name, sex and date of birth. Load these datasets into a pandas
``DataFrame``.

.. code:: python

    df_a = pandas.DataFrame(YOUR_FIRST_DATASET)
    df_b = pandas.DataFrame(YOUR_SECOND_DATASET)

Comparing all record can be computationally intensive. Therefore, we
make smart set of candidate links with one of the built-in indexing
techniques like **blocking**. Only records pairs agreeing on the
surname are included.

.. code:: python

    indexer = recordlinkage.Index()
    indexer.block('surname')
    candidate_links = indexer.index(df_a, df_b)

Each ``candidate_link`` needs to be compared on the comparable attributes.
This can be done easily with the Compare class and the available comparison
and similarity measures.

.. code:: python

    compare = recordlinkage.Compare()

    compare.string('name', 'name', method='jarowinkler', threshold=0.85)
    compare.exact('sex', 'gender')
    compare.exact('dob', 'date_of_birth')
    compare.string('streetname', 'streetname', method='damerau_levenshtein', threshold=0.7)
    compare.exact('place', 'placename')
    compare.exact('haircolor', 'haircolor', missing_value=9)

    # The comparison vectors
    compare_vectors = compare.compute(candidate_links, df_a, df_b)

This record linkage package contains several classification algorithms.
Plenty of the algorithms need trainings data (supervised learning) while
some others are unsupervised. An example of supervised learning:

.. code:: python

    true_linkage = pandas.Series(YOUR_GOLDEN_DATA, index=pandas.MultiIndex(YOUR_MULTI_INDEX))

    logrg = recordlinkage.LogisticRegressionClassifier()
    logrg.fit(compare_vectors[true_linkage.index], true_linkage)

    logrg.predict(compare_vectors)

and an example of unsupervised learning (the well known ECM-algorithm):

.. code:: python

    ecm = recordlinkage.BernoulliEMClassifier()
    ecm.fit_predict(compare_vectors)


