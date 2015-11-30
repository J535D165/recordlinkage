
.. code:: python

    %cd -q ..

The next step is to load and standardise the data to deduplicate.

.. code:: python

    import pandas
    import numpy
    
    import recordlinkage
    from recordlinkage import datasets

.. code:: python

    dfA = datasets.load_duplicated()


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-4-2291f8bfbb6c> in <module>()
    ----> 1 dfA = datasets.load_duplicated()
    

    AttributeError: 'module' object has no attribute 'load_duplicated'

