*************
Release notes
*************

Version 0.15
============

- Remove deprecated recordlinkage classes  (`#173`_)
- Bump min Python version to 3.6, ideally 3.8+ (`#171`_)
- Bump min pandas version to >=1
- Resolve deprecation warnings for numpy and pandas
- Happy lint, sort imports, format code with yapf
- Remove unnecessary np.sort in SNI algorithm (`#141`_)
- Fix bug for cosine and qgram string comparisons with threshold (`#135`_)
- Fix several typos in docs (`#151`_)(`#152`_)(`#153`_)(`#154`_)(`#163`_)(`#164`_)
- Fix random indexer (`#158`_)
- Fix various deprecation warnings and broken docs build (`#170`_)
- Fix broken docs build due to pandas depr warnings (`#169`_)
- Fix broken build and removed warning messages (`#168`_)
- Update narrative
- Replace Travis by Github Actions (`#132`_)
- Fix broken test NotFittedError
- Fix bug in low memory random sampling and add more tests (`#130`_)
- Add extras_require to setup.py for deps management
- Add banner to README and update title
- Add Binder and Colab buttons at tutorials (`#174`_)

Special thanks to Tomasz Waleń @twalen and other contributors for their
work on this release.

.. _#173: https://github.com/J535D165/recordlinkage/pull/173
.. _#171: https://github.com/J535D165/recordlinkage/pull/171
.. _#141: https://github.com/J535D165/recordlinkage/pull/141
.. _#135: https://github.com/J535D165/recordlinkage/pull/135
.. _#151: https://github.com/J535D165/recordlinkage/pull/151
.. _#152: https://github.com/J535D165/recordlinkage/pull/152
.. _#153: https://github.com/J535D165/recordlinkage/pull/153
.. _#154: https://github.com/J535D165/recordlinkage/pull/154
.. _#163: https://github.com/J535D165/recordlinkage/pull/163
.. _#164: https://github.com/J535D165/recordlinkage/pull/164
.. _#158: https://github.com/J535D165/recordlinkage/pull/158
.. _#170: https://github.com/J535D165/recordlinkage/pull/170
.. _#169: https://github.com/J535D165/recordlinkage/pull/169
.. _#168: https://github.com/J535D165/recordlinkage/pull/168
.. _#132: https://github.com/J535D165/recordlinkage/pull/132
.. _#130: https://github.com/J535D165/recordlinkage/pull/130
.. _#174: https://github.com/J535D165/recordlinkage/pull/174

Version 0.14
============

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
