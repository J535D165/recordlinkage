Comparing
=========

A set of informative, discriminating and independent features is important for
a good classification of record pairs into matching and distinct pairs. The
``recordlinkage.Compare`` class and its methods can be used to compare records
pairs. Several comparison methods are included such as string similarity
measures, numerical measures and distance measures.

.. automodule:: recordlinkage.comparing

.. autoclass:: Compare
	:members: exact, string, numeric, date, geo, compare_vectorized, compute


Migrating
---------

Version 0.10 of the Python Record Linkage Toolkit uses a new API to compare
record pairs. The new API uses a different syntax. Records are now compared
after calling the `compute` method. Also, the `Compare` class is no longer
initialized with the data and the record pairs. The data and record pairs are
passed to the `compute` method. **The old procedure still works but will be
removed in the future.**

Old (linking): 

.. code:: python

    c = recordlinkage.Compare(candidate_links, df_a, df_b)

    c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
    c.exact('sex', 'gender')
    c.date('dob', 'date_of_birth')
    c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
    c.exact('place', 'placename')
    c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

    # The comparison vectors
    c.vectors


New (linking):

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

Old (deduplication): 

.. code:: python

    c = recordlinkage.Compare(candidate_links, df_a)

    c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
    c.exact('sex', 'gender')
    c.date('dob', 'date_of_birth')
    c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
    c.exact('place', 'placename')
    c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

    # The comparison vectors
    c.vectors

New (deduplication):

.. code:: python

    c = recordlinkage.Compare()

    c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
    c.exact('sex', 'gender')
    c.date('dob', 'date_of_birth')
    c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
    c.exact('place', 'placename')
    c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

    # The comparison vectors
    feature_vectors = c.compute(candidate_links, df_a)