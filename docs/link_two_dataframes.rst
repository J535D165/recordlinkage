
Linking two census datasets
===========================

This example shows how two datasets with name and contact details
information about individuals can be linked. We will try to link the
data based on the first name, last name, sex, birthdate, city, street,
job and email. The data used in this example is fictitious.

Firstly, start with importing the ``recordlinkage`` module. The
submodule ``recordlinkage.datasets`` contains several datasets which can
be used for testing. For this example, we use the datasets ``censusA``
and ``censusB`` that can be loaded with the functions ``load_censusA``
en ``load_censusB`` respectively.

.. code:: python

    import recordlinkage
    from recordlinkage.datasets import load_censusA, load_censusB

The datasets ``censusA`` and ``censusB`` are loaded with the following
code. The datasets are ``pandas.DataFrame`` objects. This makes it easy
to manipulate the data if desired. For details about data manipulation
with ``pandas``, see their comprehensive documentation
http://pandas.pydata.org/.

.. code:: python

    dfA = load_censusA()
    dfB = load_censusB()


::


    ---------------------------------------------------------------------------

    IOError                                   Traceback (most recent call last)

    <ipython-input-2-3c88f020febd> in <module>()
    ----> 1 dfA = load_censusA()
          2 dfB = load_censusB()


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/recordlinkage-0.0.4+79.g9ef16c3.dirty-py2.7.egg/recordlinkage/datasets/__init__.pyc in load_censusA()
         28 def load_censusA():
         29 
    ---> 30         df = pd.read_csv('recordlinkage/datasets/data/personaldata1000A.csv', sep=';', index_col='record_id', encoding='utf-8')
         31         df.index.name = 'index_A'
         32         return df


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/pandas/io/parsers.pyc in parser_f(filepath_or_buffer, sep, dialect, compression, doublequote, escapechar, quotechar, quoting, skipinitialspace, lineterminator, header, index_col, names, prefix, skiprows, skipfooter, skip_footer, na_values, true_values, false_values, delimiter, converters, dtype, usecols, engine, delim_whitespace, as_recarray, na_filter, compact_ints, use_unsigned, low_memory, buffer_lines, warn_bad_lines, error_bad_lines, keep_default_na, thousands, comment, decimal, parse_dates, keep_date_col, dayfirst, date_parser, memory_map, float_precision, nrows, iterator, chunksize, verbose, encoding, squeeze, mangle_dupe_cols, tupleize_cols, infer_datetime_format, skip_blank_lines)
        496                     skip_blank_lines=skip_blank_lines)
        497 
    --> 498         return _read(filepath_or_buffer, kwds)
        499 
        500     parser_f.__name__ = name


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/pandas/io/parsers.pyc in _read(filepath_or_buffer, kwds)
        273 
        274     # Create the parser.
    --> 275     parser = TextFileReader(filepath_or_buffer, **kwds)
        276 
        277     if (nrows is not None) and (chunksize is not None):


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/pandas/io/parsers.pyc in __init__(self, f, engine, **kwds)
        588             self.options['has_index_names'] = kwds['has_index_names']
        589 
    --> 590         self._make_engine(self.engine)
        591 
        592     def _get_options_with_defaults(self, engine):


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/pandas/io/parsers.pyc in _make_engine(self, engine)
        729     def _make_engine(self, engine='c'):
        730         if engine == 'c':
    --> 731             self._engine = CParserWrapper(self.f, **self.options)
        732         else:
        733             if engine == 'python':


    /Users/jonathandebruin/anaconda/lib/python2.7/site-packages/pandas/io/parsers.pyc in __init__(self, src, **kwds)
       1101         kwds['allow_leading_cols'] = self.index_col is not False
       1102 
    -> 1103         self._reader = _parser.TextReader(src, **kwds)
       1104 
       1105         # XXX


    pandas/parser.pyx in pandas.parser.TextReader.__cinit__ (pandas/parser.c:3246)()


    pandas/parser.pyx in pandas.parser.TextReader._setup_parser_source (pandas/parser.c:6111)()


    IOError: File recordlinkage/datasets/data/personaldata1000A.csv does not exist


Making record pairs
-------------------

It is very intuitive to start with comparing each record of DataFrame
``dfA`` with all records of DataFrame ``dfB``. In fact, we want to make
record pairs. Each record pair should contain one record of ``dfA`` and
one record of ``dfB``. This process of making record pairs is also
called 'indexing'. With the ``recordlinkage`` module, indexing is easy.
Firstly, load the ``Pairs`` class. This class takes two dataframes as
input arguments. In case of deduplication of a single dataframe, one
dataframe is sufficient as input argument.

.. code:: python

    pcl = recordlinkage.Pairs(dfA, dfB)

With the method ``Pairs.full``, all possible (and unique) record pairs
are made. The method returns a ``pandas.MultiIndex``.

.. code:: python

    pairs = pcl.full()

The number of pairs is equal to the number of records in ``dfA`` times
the number of records in ``dfB``.

.. code:: python

    len(dfA)*len(dfB) == len(pairs)

Many of the record pairs do not belong to the same person. In case of
one-to-one matching, the largest number of matches should be the number
of records in the smallest dataframe. In case of full indexing,
``min(len(dfA), len(N_dfB))`` is much smaller than ``len(pairs)``. The
``recordlinkage`` module has some more advanced indexing methods to
reduce the number of record pairs. Obvious non-matches are left out of
the index. Note that if a matching record pair is not included in the
index, it can not be matched anymore.

One of the most well known indexing methods is named 'blocking'. This
method includes only record pairs that are identical on one or more
stored attributes of the person (or entity in general). The blocking
method can be used in the ``recordlinkage`` module.

.. code:: python

    pcl.block('first_name');

The argument 'first\_name' is the blocking variable. This variable has
to be the name of a column in ``dfA`` and ``dfB``. It is possible to
parse a list of columns names to block on multiple variables. Blocking
on multiple variables will reduce the number of record pairs even
further.

Another implemented indexing method is sortedneighbourhood indexing
(``Pairs.sortedneighbourhood``). This method is very useful when there
are many misspellings in the string were used for indexing. In fact,
sorted neighbourhood indexing is a generalisation of blocking. See the
documentation for details about sorted neighbourd indexing.

Comparing record pairs
----------------------

Now we now have a large set of record pairs. To compare Each record pair
is compared on some attributes both record have in common.

.. code:: python

    compare_cl = recordlinkage.Compare(pairs, dfA, dfB)
    compare_cl.exact('first_name', 'first_name', name='first_name')
    compare_cl.exact('last_name', 'last_name', name='last_name')
    compare_cl.exact('sex', 'sex', name='sex')
    compare_cl.exact('birthdate', 'birthdate', name='birthdate')
    compare_cl.exact('city', 'city', name='city')
    compare_cl.exact('street_address', 'street_address', name='street_address')
    compare_cl.exact('job', 'job', name='job')
    compare_cl.exact('email', 'email', name='email');

All comparisons are stored in a dataframe with horizontally the
comparison features and vertically the record pairs. The comparison can
be found in ``vectors`` attribute of the ``Compare`` class. The first 10
comparison vectors are:

.. code:: python

    compare_cl.vectors.head(10)

.. code:: python

    ecm_cl = recordlinkage.ExpectationMaximisationClassifier(method='ecm')
    
    ecm_cl.learn(compare_cl.vectors)
