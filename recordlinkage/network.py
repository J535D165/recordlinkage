import pandas as pd

from recordlinkage.types import is_pandas_2d_multiindex
from recordlinkage.types import is_pandas_multiindex


class OneToOneLinking:
    """[EXPERIMENTAL] One-to-one linking

    A record from dataset A can match at most one record from dataset
    B. For example, (a1, a2) are records from A and (b1, b2) are records
    from B.  A linkage of (a1, b1), (a1, b2), (a2, b1), (a2, b2) is not
    one-to-one  connected. One of the results of one-to-one linking can
    be (a1, b1), (a2, b2).

    Parameters
    ----------
    method : str
        The method to solve the problem. Only 'greedy' is supported at
        the moment.

    Note
    ----

    This class is experimental and might change in future versions.

    """

    def __init__(self, method="greedy"):
        super().__init__()

        self.method = method

    @classmethod
    def _bool_duplicated(cls, links, level):
        return links.get_level_values(level).duplicated()

    def _compute_greedy(self, links):
        result = []
        set_a = set()
        set_b = set()

        for index_a, index_b in links:
            if index_a not in set_a and index_b not in set_b:
                result.append((index_a, index_b))
                set_a.add(index_a)
                set_b.add(index_b)

        return pd.MultiIndex.from_tuples(result)

    def _compute(self, links):
        if not is_pandas_2d_multiindex(links):
            if not is_pandas_multiindex(links):
                raise TypeError("expected pandas.MultiIndex")
            elif not is_pandas_2d_multiindex(links):
                raise ValueError(
                    "pandas.MultiIndex has incorrect number of "
                    "levels (expected 2 levels)"
                )

        if self.method == "greedy":
            return self._compute_greedy(links)
        else:
            raise ValueError(f"unknown matching method {self.method}")

    def compute(self, links):
        """Compute the one-to-one linking.

        Parameters
        ----------
        links : pandas.MultiIndex
            The pairs to apply linking to.

        Returns
        -------
        pandas.MultiIndex
            A one-to-one matched MultiIndex of record pairs.

        """

        return self._compute(links)


class OneToManyLinking(OneToOneLinking):
    """[EXPERIMENTAL] One-to-many linking

    A record from dataset A can link multiple records from dataset B,
    but a record from B can link to only one record of dataset A. Use
    the `level` argument to switch A and B.

    Parameters
    ----------
    level : int
        The level of the MultiIndex to have the one relations. The
        options are 0 or 1 (incication the level of the MultiIndex).
        Default 0.
    method : str
        The method to solve the problem. Only 'greedy' is supported at
        the moment.

    Example
    -------

    Consider a MultiIndex with record pairs constructed from datasets A
    and B. To link a record from B to at most one record of B, use the
    following syntax:

    > one_to_many = OneToManyLinking(0)
    > one_to_many.compute(links)

    To link a record from B to at most one record
    of B, use:

    > one_to_many = OneToManyLinking(1)
    > one_to_many.compute(links)

    Note
    ----

    This class is experimental and might change in future versions.

    """

    def __init__(self, level=0, method="greedy"):
        super().__init__(method=method)

        self.level = level

    def _compute_greedy(self, links):
        source_dupl_bool = self._bool_duplicated(links, self.level)
        return links[~source_dupl_bool]

    def compute(self, links):
        """Compute the one-to-many matching.

        Parameters
        ----------
        links : pandas.MultiIndex
            The pairs to apply linking to.

        Returns
        -------
        pandas.MultiIndex
            A one-to-many matched MultiIndex of record pairs.

        """

        return self._compute(links)


class ConnectedComponents:
    """[EXPERIMENTAL] Connected record pairs

    This class identifies connected record pairs. Connected components
    are especially used in detecting duplicates in a single dataset.

    Note
    ----

    This class is experimental and might change in future versions.
    """

    def __init__(self):
        super().__init__()

    def compute(self, links):
        """Return the connected components.

        Parameters
        ----------
        links : pandas.MultiIndex
            The links to apply one-to-one matching on.

        Returns
        -------
        list of pandas.MultiIndex
            A list with pandas.MultiIndex objects. Each MultiIndex
            object represents a set of connected record pairs.

        """

        try:
            import networkx as nx
        except ImportError():
            raise Exception("'networkx' module is needed for this operation")

        graph_pairs = nx.Graph()
        graph_pairs.add_edges_from(links.values)
        connected_pairs = (
            graph_pairs.subgraph(c).copy() for c in nx.connected_components(graph_pairs)
        )

        links_result = [
            pd.MultiIndex.from_tuples(subgraph.edges()) for subgraph in connected_pairs
        ]

        return links_result
