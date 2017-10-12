Conflict Resolution
===================

The ``recordlinkage.algorithms.conflict_resolution`` module
contains a large number of conflict resolution functions.
These functions can be used with ``recordlinkage.FuseLinks.resolve``
if a conflict handling strategy is needed, which is not currently
implemented in the ``recordlinkage.FuseLinks`` interface.

These conflict resolution functions are based on:
Bleiholder, J., & Naumann, F. (2006). Conflict Handling Strategies in an Integrated Information System.
Humboldt-Universität zu Berlin, Mathematisch-Naturwissenschaftliche Fakultät II, Institut für Informatik.
http://doi.org/http://dx.doi.org/10.18452/2460

.. automodule:: recordlinkage.algorithms.conflict_resolution

.. autofunction:: aggregate

.. autofunction:: annotated_concat

.. autofunction:: choose_first

.. autofunction:: choose_last

.. autofunction:: choose_longest

.. autofunction:: choose_longest_tie_break

.. autofunction:: choose_max

.. autofunction:: choose_metadata_max

.. autofunction:: choose_metadata_min

.. autofunction:: choose_min

.. autofunction:: choose_random

.. autofunction:: choose_shortest

.. autofunction:: choose_shortest_tie_break

.. autofunction:: choose_trusted

.. autofunction:: count

.. autofunction:: group

.. autofunction:: identity

.. autofunction:: no_gossip

.. autofunction:: nullify

.. autofunction:: vote
