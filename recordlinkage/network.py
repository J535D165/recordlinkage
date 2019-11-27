import networkx as nx
import pandas as pd
from recordlinkage.types import (
    is_pandas_2d_multiindex,
    is_pandas_like,
    is_pandas_multiindex,
)


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
        The method to solve the problem. The options are 'greedy' and "max_weighted". The "max_weighted" option solves the assignment problem, i.e. it finds the one to one matching with the greatest combined weight of all links. The matching is done with the Blossom algorithm by Jack Edmonds as implemented in networkx. For more details, see https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html.

    Note
    ----

    This class is experimental and might change in future versions.

    """

    def __init__(self, method="greedy"):
        super(OneToOneLinking, self).__init__()

        self.method = method

    def add_weights(self, links, features=None, classifier=None, method="weights"):
        """Add match weights to the candidate matches.

        Parameters
        ----------
        links : pandas.MultiIndex
            The candidate matches as calculated by a recordlinkage classifier.
        features : pandas.DataFrame
            The dataframe with similarity weights as calculated by a recordlinkage.Compare object.
        classifier : recordlinkage.base.Classifier
            The classifier used to classify the records in matches and non-matches.
        method : str
            The method to assign weights to the candidate matces. The options are 'weights', 'log_weights' and 'probabilities'. The 'weights' features requires the features to be passed. It adds the sum of the similarity weights from features to the links. Both 'log_weights' and 'probabilities' require that the classifier is passed. 'log_weights' adds the matching weight as defined in the Fellegi-Sunter framework. These weights can be negative, but the "max_weighted" linking strategy can't handle negative weights. All matching weights are offset with the largest total negative matching weight, so all the weights are greater than or equal to 0. This method is only available for the ECM and NaiveBayes classifier. 'probabilities' adds the probabilities that the record pair is a match as a weight. This method is available for every classifier.

        Returns
        -------
        pandas.Series

        Example
        -------

        Consider a MultiIndex with record pairs constructed from datasets A
        and B. The candidate matches are determined with a classifier. To link a candidate match from A to at most one record of B with the "max_weighted" method, weights need to be added to the candidate matches. This can be done by using the features or the classifier. Given the following set up:

            > indexer = Index()
            > indexer.full()
            > record_pairs = indexer.index(A, B)
            > comparator = Compare(compare.String("A_string", "B_string"))
            > features = comparator.compute(record_pairs, A, B)
            > ecm = ECMClassifier()
            > candidates = ecm.fit_predict(features)
            > one_to_one = OneToOneLinking(method="max_weighted")

        Weights can be added with the following syntax:

            > candidates_weights = one_to_one.add_weights(candidates, features=features, method="weights")
            > candidates_log_weights = one_to_one.add_weights(candidates, classifier=ecm, method="log_weights")

        """

        # get subset of data that correponds with the multiindex links
        difference = features.index.difference(links)
        features = features.drop(index=difference)

        if method == "weights" or method == "log_weights":
            initial_columns = features.columns

            if method == "weights":
                weight = features.sum(axis=1)

            elif method == "log_weights":
                # calculate the total log weight for each row
                weight = pd.Series(0, index=features.index)
                for column, weights in classifier.log_weights.items():
                    weight += features[column].apply(lambda x: weights[x])

                # offset negative values
                min_weight = weight.min()
                if min_weight < 0:
                    weight = weight - min_weight

            #  add the weight and remove all other columns
            features = features.assign(weight=weight)
            links = features.drop(columns=initial_columns).squeeze()

        elif method == "probabilities":

            links = classifier.prob(features)

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

    def _compute_max_weighted(self, links):
        """Compute a one to one linking by maximizing the total similarity weight."""

        graph = self._to_weighted_bipartite_graph(links)
        max_weighted_graph = self._max_weighted_graph(graph)
        max_weighted_series = self._to_max_weighted_series(max_weighted_graph)

        return max_weighted_series

    def _max_weighted_graph(self, graph):
        """Calculate the maximally weighted bipartite graph with the Blossom algorithm by Edmonds."""

        # max weight matching
        max_weighted_edges = nx.algorithms.matching.max_weight_matching(graph)

        # restore order after matching
        max_weighted_edges = self._order_max_weighted_bipartite_graph(
            graph, max_weighted_edges
        )

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

    def _compute(self, links):

        if self.method == "greedy":
            if not is_pandas_multiindex(links):
                raise TypeError("expected pandas.MultiIndex")
            if not is_pandas_2d_multiindex(links):
                raise ValueError(
                    "pandas.MultiIndex has incorrect number of "
                    "levels (expected 2 levels)"
                )

            return self._compute_greedy(links)

        elif self.method == "max_weighted":
            if not is_pandas_like(links):
                raise TypeError(
                    "expected pandas.Series with a MultiIndex and weights as values"
                )
            if not is_pandas_2d_multiindex(links.index):
                raise ValueError(
                    "pandas.MultiIndex has incorrect number of "
                    "levels (expected 2 levels)"
                )
            return self._compute_max_weighted(links)

        else:
            raise ValueError("unknown matching method {}".format(self.method))

    def _order_max_weighted_bipartite_graph(self, graph, max_weighted_edges):
        """Swaps the order of edges that are swapped after max weight matching."""

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
        """Convert a Series with MultiIndex and weights to a bipartite graph with weighted edges."""

        # don't change the passed series
        tmp = links.copy()
        # add labels to both multiindex levels to ensure no overlap of nodes in the graph
        tmp.index = self._add_node_labels_to_multiindex(tmp.index)

        # create the graph
        graph = nx.Graph()

        left = tmp.index.levels[0]
        right = tmp.index.levels[1]
        values = tmp.values

        graph.add_nodes_from(left, bipartite=0)
        graph.add_nodes_from(right, bipartite=1)

        graph.add_weighted_edges_from(list(zip(left, right, values)))

        return graph

    def _to_max_weighted_series(self, graph):
        """Convert a (max weighted) bipartite graph to a Series."""

        max_weighted_series = nx.to_pandas_edgelist(graph)

        # ensure output format is the same as the format of the initial candidate links
        max_weighted_series = max_weighted_series.set_index(
            ["source", "target"]
        ).squeeze()
        max_weighted_series.index.names = [None, None]
        max_weighted_series.index = self._remove_node_labels_from_multiindex(
            max_weighted_series.index
        )

        return max_weighted_series

    def _add_node_labels_to_multiindex(self, multiindex, labels=["left_", "right_"]):
        """Adds labels to a MultiIndex. This is done in order to distinguish the left and right dataset during the max weighted matching algorithm."""

        for i, (level, dataset) in enumerate(zip(multiindex.levels, labels)):
            stringified_level = [dataset + str(value) for value in level]
            multiindex = multiindex.set_levels(stringified_level, i)

        return multiindex

    def _remove_node_labels_from_multiindex(
        self, multiindex, labels=["left_", "right_"]
    ):

        for i, (level, label) in enumerate(zip(multiindex.levels, labels)):
            destringified_level = [int(value.replace(label, "")) for value in level]
            multiindex = multiindex.set_levels(destringified_level, i)

        return multiindex

    def compute(self, links):
        """Compute the one-to-one linking.

        Parameters
        ----------
        links : pandas.MultiIndex or pandas.Series
            The pairs to apply linking to. Should be a pandas.MultiIndex for the 'greedy' and a pandas.Series for the 'max_weighted' method.

        Returns
        -------
        pandas.MultiIndex or pandas.Series
            A one-to-one matched MultiIndex of record pairs for the 'greedy' method and a pandas.Series with one-to-one matched record pairs and their matching weight.

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
        super(OneToManyLinking, self).__init__(method=method)

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

        links_result = [
            pd.MultiIndex.from_tuples(subgraph.edges())
            for subgraph in connected_components
        ]

        return links_result
