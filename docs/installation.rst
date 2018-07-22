******************
Installation guide
******************

Python version support
======================

The Python Record Linkage Toolkit supports the versions of Python that Pandas
supports as well. You can find the supported Python versions in the Pandas
documentation_.

.. _documentation: http://pandas.pydata.org/pandas-docs/stable/install.html#python-version-support

Installation
============

The easiest way of installing the Python Record Linkage Toolkit is using
``pip``. It is as easy as typing:

.. code:: sh

	pip install --user recordlinkage

You can also clone the project on Github. The license of this record linkage
package is BSD-3-Clause.

Dependencies
============

The following packages are required. You probably have most of it already ;)

-  `numpy <http://www.numpy.org>`__
-  `pandas (>=0.18.0) <https://github.com/pydata/pandas>`__
-  `scipy <https://www.scipy.org/>`__
-  `sklearn <http://scikit-learn.org/>`__
-  `jellyfish <https://github.com/jamesturk/jellyfish>`__: Needed for
   approximate string comparison and string encoding. 
-  `numexpr (optional) <https://github.com/pydata/numexpr>`__: Used to speed up 
   numeric comparisons. 



