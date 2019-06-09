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

The Python Record linkage Toolkit requires Python 3.5 or higher (since version
>= 0.14). Install the package easily with pip

.. code:: sh

    pip install recordlinkage

Python 2.7 users can use version <= 0.13, but it is advised to use Python >=
3.5.

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



