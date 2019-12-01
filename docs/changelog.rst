*************
Release notes
*************

Version 0.14.0
==============

- Drop Python 2.7 and Python 3.4 support. (`#91`_) 
- Upgrade minimal pandas version to 0.23.
- Simplify the use of all cpus in parallel mode. (`#102`_)
- Store large example datasets in user home folder or use environment 
  variable. Before, example datasets were stored in the package. (see 
  issue `#42`_) (`#92`_)
- Add support to write and read annotation files for recordlinkage ANNOTATOR.
  See the docs and https://github.com/J535D165/recordlinkage-annotator for 
  more information.
- Replace `.labels` by `.codes` for `pandas.MultiIndex` objects for newer
  versions of pandas (>0.24).  (`#103`_) 
- Fix totals for pandas.MultiIndex input on confusion matrix and accuracy 
  metrics. (see issue `#84`_) (`#109`_)
- Initialize Compare with (a list of) features (Bug). (`#124`_)
- Various updates in relation to deprecation warnings in third-party 
  libraries such as sklearn, pandas and networkx.

.. _#42: https://github.com/J535D165/recordlinkage/issues/42
.. _#84: https://github.com/J535D165/recordlinkage/issues/84

.. _#91: https://github.com/J535D165/recordlinkage/pull/91
.. _#92: https://github.com/J535D165/recordlinkage/pull/92
.. _#102: https://github.com/J535D165/recordlinkage/pull/102
.. _#103: https://github.com/J535D165/recordlinkage/pull/103
.. _#109: https://github.com/J535D165/recordlinkage/pull/109
.. _#124: https://github.com/J535D165/recordlinkage/pull/124
