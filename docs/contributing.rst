************
Contributing
************

Thanks for your interest in contributing to the Python Record Linkage Toolkit.
There is a lot of work to do. See `Github <https://github.com/J535D165/recordlinkage/graphs/contributors>`_ 
for the contributors to this package. 

The workflow for contributing is as follows:

- clone https://github.com/J535D165/recordlinkage.git
- Make a branch with your modifications/contributions
- Write tests
- Run all tests
- Do a pull request

Testing
=======

Install `pytest`:

.. code:: sh

	pip install pytest

Run the following command to test the package

.. code:: sh

	python -m pytest tests/

Performance
===========

Performance is very important in record linkage. The performance is monitored
for all serious modifications of the core API. The performance monitoring is
performed with `Airspeed Velocity <http://github.com/spacetelescope/asv/>`_
(asv).

Install Airspeed Velocity:

.. code:: sh

	pip install asv

Run the following command from the root of the repository to test the
performance of the current version of the package:

.. code:: sh

	asv run

Run the following command to test all versions since tag v0.6.0

.. code:: sh

	asv run --skip-existing-commits v0.6.0..master






