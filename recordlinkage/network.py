import pandas as pd
import networkx as nx

from recordlinkage.types import is_pandas_2d_multiindex
from recordlinkage.types import is_pandas_multiindex
from recordlinkage.types import is_pandas_like


class OneToOneLinking(object):
    """[EXPERIMENTAL] One-to-one linking

    A record from dataset A can match at most one record from dataset
    B. For example, (a1, a2) are records from A and (b1, b2) are records
    from B.  A linkage of (a1, b1), (a1, b2), (a2, b1), (a2, b2) is not
    one-to-one  connected. One of the results of one-to-one linking can
    be (a1, b1), (a2, b2).

    Parameters
    ----------
    method : str
        The method to solve the problem. The options are 'greedy' and 'max_weighted'.

    Note
    ----

    This class is experimental and might change in future versions.

    """

    def __init__(self, method='greedy'):
        super(OneToOneLinking, self).__init__()

        self.method = method

    def _add_similarity_weights(self, links, data):
        """Add the similarity weights to the MultiIndex with candidate links."""

        #  calculate the total weight and remove all other columns
        initial_columns = data.columns
        data["weight"] = data.sum(axis=1)

        # slicing on a multiindex is equivalent to merging on two columns
        data = data.drop(columns=initial_columns).reset_index()
        links = links.to_frame(index=False).rename(columns={0: "level_0", 1: "level_1"})
        links = links.merge(data, how="left", on=["level_0", "level_1"]).set_index(["level_0", "level_1"])
        links.index.names = [None, None]

        return links

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

    def _compute_max_weighted(self, links, data):
        """Compute a one to one linking by maximizing the total similarity weight."""

        links = self._add_similarity_weights(links, data)
        graph = self._to_weighted_bipartite_graph(links)

        max_weighted_graph = self._max_weighted_graph(graph)
        max_weighted_dataframe = self._to_max_weighted_dataframe(max_weighted_graph)

        return max_weighted_dataframe

    def _max_weighted_graph(self, graph):
        """Calculate the maximally weighted bipartite graph."""

        # max weight matching
        max_weighted_edges = nx.algorithms.matching.max_weight_matching(graph)

        # restore order after matching
        max_weighted_edges = self._order_max_weighted_bipartite_graph(graph, max_weighted_edges)

        # create maximally weighted graph
        weights = [graph[u][v]["weight"] for u, v in max_weighted_edges]
        max_weighted_left = [edge[0] for edge in max_weighted_edges]
        max_weighted_right = [edge[1] for edge in max_weighted_edges]

        max_weighted_graph = nx.Graph()

        max_weighted_graph.add_nodes_from(max_weighted_left, bipartite=0)
        max_weighted_graph.add_nodes_from(max_weighted_right, bipartite=1)
        max_weighted_graph.add_weighted_edges_from(
            list(zip(max_weighted_left, max_weighted_right, weights))
        )

        return max_weighted_graph

    def _compute(self, links, data):
        if not is_pandas_2d_multiindex(links):
            if not is_pandas_multiindex(links):
                raise TypeError("expected pandas.MultiIndex")
            elif not is_pandas_2d_multiindex(links):
                raise ValueError(
                    "pandas.MultiIndex has incorrect number of "
                    "levels (expected 2 levels)")
        if (data is not None) and (not is_pandas_like(data)):
            raise TypeError("expected pandas.DataFrame")

        if self.method == "greedy":
            return self._compute_greedy(links)
        elif self.method == "max_weighted":
            return self._compute_max_weighted(links, data)
        else:
            raise ValueError("unknown matching method {}".format(self.method))

    def _order_max_weighted_bipartite_graph(self, graph, max_weighted_edges):
        """Swaps the order of edges that are swapped after max. weight matching."""

        edges_left = list(set(edge[0] for edge in graph.edges))

        max_weighted_left = [edge[0] for edge in max_weighted_edges]
        max_weighted_right = [edge[1] for edge in max_weighted_edges]

        for i, value in enumerate(max_weighted_left):
            if value not in edges_left:
                max_weighted_left[i], max_weighted_right[i] = (
                    max_weighted_right[i],
                    max_weighted_left[i],
                )

        ordered_max_weighted_edges = list(zip(max_weighted_left, max_weighted_right))

        return ordered_max_weighted_edges

    def _to_weighted_bipartite_graph(self, links):
        """Convert a pandas DataFrame with MultiIndex and single column weight to a bipartite graph with weighted edges."""

        # add labels to both multiindex levels to ensure no overlap of nodes in the graph
        links = links.set_index(self._add_node_labels_to_multiindex(links.index))
        links = links.reset_index()

        # create the graph
        graph = nx.Graph()

        graph.add_nodes_from(links["level_0"], bipartite=0)
        graph.add_nodes_from(links["level_1"], bipartite=1)

        graph.add_weighted_edges_from(
            list(zip(links["level_0"], links["level_1"], links["weight"]))
        )

        return graph

    def _to_max_weighted_dataframe(self, graph):
        """Convert a (max weighted) bipartite graph to a DataFrame."""

        max_weighted_dataframe = nx.to_pandas_edgelist(graph)

        # ensure output format is the same as the format of the initial candidate links
        max_weighted_dataframe = max_weighted_dataframe.set_index(["source", "target"])
        max_weighted_dataframe.index.names = [None, None]
        max_weighted_dataframe = max_weighted_dataframe.set_index(
            self._remove_node_labels_from_multiindex(max_weighted_dataframe.index)
        ).sort_index(level=0)

        return max_weighted_dataframe

    def _add_node_labels_to_multiindex(self, multiindex, labels=["left_", "right_"]):

        for i, (level, dataset) in enumerate(zip(multiindex.levels, labels)):
            stringified_level = [dataset + str(value) for value in level]
            multiindex = multiindex.set_levels(stringified_level, i)

        return multiindex

    def _remove_node_labels_from_multiindex(self, multiindex, labels=["left_", "right_"]):

        for i, (level, label) in enumerate(zip(multiindex.levels, labels)):
            destringified_level = [int(value.replace(label, "")) for value in level]
            multiindex = multiindex.set_levels(destringified_level, i)

        return multiindex

    def compute(self, links, data=None):
        """Compute the one-to-one linking.

        Parameters
        ----------
        links : pandas.MultiIndex
            The pairs to apply linking to.
        data : pandas.DataFrame
            The similary weights computed for the entire dataset.

        Returns
        -------
        pandas.MultiIndex
            A one-to-one matched MultiIndex of record pairs.

        """

        return self._compute(links, data)


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

    def __init__(self, level=0, method='greedy'):
        super(OneToManyLinking, self).__init__(method=method)

        self.level = level

    def _compute_greedy(self, links):

        source_dupl_bool = self._bool_duplicated(links, self.level)
        return links[~source_dupl_bool]

    def compute(self, links, data=None):
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

        return self._compute(links, data)


class ConnectedComponents(object):
    """[EXPERIMENTAL] Connected record pairs

    This class identifies connected record pairs. Connected components
    are especially used in detecting duplicates in a single dataset.

    Note
    ----

    This class is experimental and might change in future versions.
    """

    def __init__(self):
        super(ConnectedComponents, self).__init__()

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

        G = nx.Graph()
        G.add_edges_from(links.values)
        connected_components = nx.connected_component_subgraphs(G)

        links_result = [pd.MultiIndex.from_tuples(subgraph.edges())
                        for subgraph in connected_components]

        return links_result
