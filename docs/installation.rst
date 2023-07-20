************
Installation
************

Python version support
======================

The Python Record Linkage Toolkit supports the versions of Python that Pandas
supports as well. You can find the supported Python versions in the Pandas
documentation.

Installation
============

The Python Record linkage Toolkit requires Python 3.6 or higher. Install the
package easily with pip

.. code:: sh

    pip install recordlinkage

You can also clone the project on Github.

To install all recommended and optional dependencies, run

.. code:: sh

    pip install recordlinkage['all']

Dependencies
============

The Python Record Linkage Toolkit depends on the following packages:

-  `numpy <http://www.numpy.org>`__
-  `pandas <https://github.com/pydata/pandas>`__
-  `scipy <https://www.scipy.org/>`__
-  `sklearn <http://scikit-learn.org/>`__
-  `jellyfish <https://github.com/jamesturk/jellyfish>`__
- `joblib`

Recommended dependencies
------------------------

-  `numexpr <https://github.com/pydata/numexpr>`__ - accelerating certain numerical operations
-  `bottleneck <https://github.com/pydata/bottleneck>`__ - accelerating certain types of nan evaluations

Optional dependecies
--------------------

- networkx - for network operations like connected components



