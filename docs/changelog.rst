*************
Release notes
*************

Version 0.11.0
==============

- The submodule 'standardise' is renamed. The new name is 'preprocessing'.
  The submodule 'standardise' will get deprecated in a next version.
- Deprecation errors were not visible for many users. In this version, the 
  errors are better visible. 
- Improved and new logs for indexing, comparing and classification.
- Faster comparing of string variables. Thanks Joel Becker. 
- Changes make it possible to pickle Compare and Index objects. This makes it
  easier to run code in parallel. Tests were added to ensure that pickling
  remains possible.
- Important change. MultiIndex objects with many record pairs were split into 
  pieces to lower memory usage. In this version, this automatic splitting is 
  removed. Please split the data yourself.
- Integer indexing. Blog post will follow on this.
- The metrics submodule has changed heavily. This will break with the previous
  version. 
- repr() and str() will return informative information for index and compare 
  objects. 
- It is possible to use abbreviations for string similarity methods. For example
  'jw' for the Jaro-Winkler method.
- The FEBRL dataset loaders can now return the true links as a 
  pandas.MultIndex for each FEBRL dataset. This option is disabled by default.
  See the [FEBRL datasets][febrl_datasets] for details. 
- Fix issue with automatic recognision of license on Github.
- Various small improvements.

[febrl_datasets]: http://recordlinkage.readthedocs.io/en/latest/ref-datasets.html#recordlinkage.datasets.load_febrl1

Note: In the next release, the Pairs class will get removed. Migrate now.

Version 0.10.1
==============

- print statement in the geo compare algorithm removed. 
- String, numeric and geo compare functions now raise directly when an
  incorrect algorithm name is passed. 
- Fix unit test that failed on Python 2.7. 

Version 0.10.0
==============

- A new compare API. The new Compare class no longer takes the datasets and 
  pairs as arguments. The actual computation is now performed when calling
  `.compute(PAIRS, DF1, DF2)`. The documentation is updated as well, but 
  still needs improvement.
- Two new string similarity measures are added: Smith Waterman 
  (smith_waterman) and Longest Common Substring (lcs). Thanks to Joel Becker 
  and Jillian Anderson from the Networks Lab of the University of Waterloo. 
- Added and/or updated a large amount of unit tests.
- Various small improvements.

Version 0.9.0
=============

- A new index API. The new index API is no longer a single class 
  (``recordlinkage.Pairs(...)``) with all the functionality in it. The new API
  is based on Tensorflow and FEBRL. With the new structure, it easier to 
  parallise the record linkage process. In future releases, this will be 
  implemented natively. `See the reference page for more information and migrating. <http://recordlinkage.readthedocs.io/en/latest/ref-index.html>`_
- Significant speed improvement of the Sorted Neighbourhood Indexing 
  algorithm. Thanks to @perryvais (PR #32). 
- The function ``binary_comparisons`` is renamed. The new name of the function
  is ``binary_vectors``. Documentation added to RTD. 
- Added unit tests to test the generation of random comparison vectors. 
- Logging module added to separate module logs from user logs. The 
  implementation is based on Tensorflow.

Version 0.8.1
=============

- Issues solved with rendering docs on ReadTheDocs. Still not clear what is 
  going on with the ``autodoc_mock_imports`` in the sphinx conf.py file. Maybe
  a bug in sphinx. 
- Move six to dependencies. 
- The reference part of the docs is split into separate subsections. This 
  makes the reference better readable. 
- The landing page of the docs is slightly changed. 

Version 0.8.0
=============

- Add additional arguments to the function that downloads and loads the 
  krebsregister data. The argument ``missing_values`` is used to fill missing
  values. Default: nothing is done. The argument ``shuffle`` is used to 
  shuffle the records. Default is True.
- Remove the lastest traces of the old package name. The new package name is
  'Python Record Linkage Toolkit'
- Better error messages when there are only matches or non-matches are passed
  to train the classifier. 
- Add AirSpeedVelocity tests to test the performance. 
- Compare for deduplication fixed. It was broken.
- Parameterized tests for the ``Compare`` class and its algorithms. Making use
  of ``nose-parameterized`` module. 
- Update documentation about contributing. 
- Bugfix/improvement when blocking on multiple columns with missing values.
- Fix bug #29 (https://github.com/J535D165/recordlinkage/issues/29). Package 
  not working with pandas 0.18 and 0.17. Dropped support pandas 0.17 and fixed
  support for 0.18. Also added multi-dendency tests for TravisCI.
- Support for dedicated deduplication algorithms 
- Special algorithm for full index in case of finding duplicates. Performce is
  100x better. 
- Function ``max_number_of_pairs`` to get the maximum number of pairs.
- ``low_memory`` for compare class. 
- Improved performance in case of comparing a large number of record pairs. 
- New documentation about custom algorithms
- New documentation about the use of classifiers. 
- Possible to compare arrays and series directly without using labels. 
- Make a dataframe with random comparison vectors with the 
  ``binary_comparisons`` in the ``recordlinkage.datasets.random`` module.
- Set KMeans cluster centers by hand. 
- Various documentation updates and improvements.
- Jellyfish is now a required dependency. Fixes bug #30 (
  https://github.com/J535D165/recordlinkage/issues/30). 
- Added ``tox.ini`` to test packaging and installation of package.
- Drop requirements.txt file. 
- Many small fixes and changes. Most of the changes cover the ``Compare`` 
  module. Especially label handling is improved. 

Version 0.7.2
=============

- Incorrect name of the Levenshtein method in the string comparison method
  fixed.

Version 0.7.1
=============

- Fix the generation of docs on ReadTheDocs.
- Installation issue fixed. Packages not found. 
- Import issues solved.

Version 0.7.0
=============

- Rename the package into 'Python Record Linkage Toolkit'
- Remove ``similar_values`` function
- Remove gender imputation tool
- Updated algorithms for comparing numberic variables. The new algorithms can
  compute the similarity with kernels like gaussian, linear, squared and 
  exponential. Tests for these numeric comparison algorithms are included. 
- Better NaN handling for compare functions.
- Algorithm added to compare dates.
- Add tests for date comparing.
- Divide the ``Compare`` class into two classes.
- Add documentation about performance tricks and concepts.
- Replace the comparison algorithms to a submodule. 
- Include six in the package
- Drop ``requests`` module and use builtin Python functions. 
- Add metaphone phonetic algorithm.
- Add match rating string comparing algorithm.
- Manual parameter handling for logistic regression. The attributes are
  ``coefficients`` and ``intercept``.
- Drop class ``BernoulliNBClassifier``.
- Various documentation updates.
- Many small other updates.

Version 0.6.0
=============

- Reformatting the code such that it follows PEP8.
- Add Travis-CI and codecov support.
- Switch to distributing wheels.
- Fix bugs with depreciated pandas functions. ``__sub__`` is no longer used
  for computing the difference of Index objects. It is now replaced by
  ``INDEX.difference(OTHER_INDEX)``.
- Exclude pairs with NaN's on the index-key in Q-gram indexing.
- Add tests for krebsregister dataset.
- Fix Python3 bug on krebsregister dataset.
- Improve unicode handling in phonetic encoding functions.
- Strip accents with the ``clean`` function.
- Add documentation
- Bug for random indexing with incorrect arguments fixed and tests added.
- Improved deployment workflow
- And much more

Version 0.5.0
=============

- Batch comparing added. Signifant speed improvement.
- rldatasets are now included in the package itself.
- Added an experimental gender imputation tool. 
- Blocking and SNI skip missing values
- No longer need for different index names
- FEBRL datasets included
- Unit tests for indexing and comparing improved
- Documentation updated

Version 0.4.0
=============

- Fixes a serious bug with deduplication.
- Fixes undesired behaviour for sorted neighbourhood indexing with missing 
  values.
- Add new datasets to the package like Febrl datasets
- Move Krebsregister dataset to this package. 
- Improve and add some tests
- Various documentation updates 

Version 0.3.0
=============

- Total restructure of compare functions (The end of changing the API is close
  to now.)
- Compare method ``numerical`` is now named ``numeric`` and ``fuzzy`` is now 
  named ``string``.
- Add haversine formula to compare geographical records. 
- Use numexpr for computing numeric comparisons.
- Add step, linear and squared comparing.
- Add eye index method.
- Improve, update and add new tests.
- Remove iterative indexing functions. 
- New add chunks for indexing functions. These chunks are defined in the class 
  Pairs. If chunks are defined, then the indexing functions returns a generator
  with an Index for each element.
- Update documentation.
- Various bug fixes.

Version 0.2.0
=============

- Full Python3 support
- Update the parameters of the Logistic Regression Classifier manually. In 
  literature, this is often denoted as the 'deterministic record linkage'.
- Expectation/Conditional Maximization algorithm completely rewritten. The 
  performance of the algorithm is much better now. The algorithm is still 
  experimental.
- New string comparison metrics: Q-gram string comparing and Cosine string
  comparing. 
- New indexing algorithm: Q-gram indexing.
- Several internal tests.
- Updated documentation.
- BernoulliNBClassifier is now named NaiveBayesClassifier. No changes to the 
  algorithm.
- Arguments order in compare functions corrected.
- Function to clean phone numbers
- Return the result of the classifier as index, numpy array or pandas series. 
- Many bug fixes

Version 0.1.0
=============
- Official release