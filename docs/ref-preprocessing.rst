****************
0. Preprocessing
****************

Preprocessing data, like cleaning and standardising, may increase your record
linkage accuracy. The Python Record Linkage Toolkit contains several tools for
data preprocessing. The preprocessing and standardising functions are
available in the submodule `recordlinkage.preprocessing`. Import the
algorithms in the following way:

.. code:: python

    from recordlinkage.preprocessing import clean, phonetic

Cleaning
========

The Python Record Linkage Toolkit has some cleaning function from which
:func:`recordlinkage.preprocessing.clean` is the most generic function. Pandas
itself is also very usefull for (string) data cleaning. See the pandas
documentation on this topic: `Working with Text Data <https://pandas.pydata.org/pandas-docs/stable/text.html>`_.

.. autofunction:: recordlinkage.preprocessing.clean
.. autofunction:: recordlinkage.preprocessing.phonenumbers
.. autofunction:: recordlinkage.preprocessing.value_occurence

Phonetic encoding
=================

Phonetic algorithms are algorithms for indexing of words by their
pronunciation. The most well-known algorithm is the `Soundex
<https://en.wikipedia.org/wiki/Soundex>`_ algorithm. The Python Record Linkage
Toolkit supports multiple algorithms through the
:func:`recordlinkage.preprocessing.phonetic` function.

.. note::

    Use phonetic algorithms in advance of the indexing and comparing step.
    This results in most siutations in better performance.

.. autofunction:: recordlinkage.preprocessing.phonetic
.. autoattribute:: recordlinkage.preprocessing.phonetic_algorithms
