# import sys
# import distutils

# import pandas as pd
# import numpy as np

# import networkx as nx
# from networkx.algorithms import bipartite


# class ManualReview:

# 	def __init__(self, dfA, dfB=None):

# 		self.A = dfA
# 		self.B = dfB

# 		# self.graph_pairs = nx.Graph(self.pairs.index)

# 	def review_pairs(pairs):

# 		print "Stop the review process with the command [stop]"

# 		# Lists with matches, nonmatches and possible matches. 
# 		match_ids = []
# 		nonmatch_ids = []
# 		possible_match_ids = []

# 		for pair in pairs:

# 			if self.B:
# 				print self.A.ix[pair.index[0]]
# 				print self.B.ix[pair.index[1]]
# 			else:
# 				print self.A.ix[[pair.index[0], pair.index[1]]]

# 			print "Do these records belong to the same entity? [y/n/?]"
# 			sys.stdout.flush()
# 			input_value = raw_input()

# 			try:
# 				same_or_not = distutils.util.strtobool(input_value)

# 				if same_or_not == 1:
# 					match_ids.append(pair.index)
# 					print "Appended to matches"
# 				else:
# 					nonmatch_ids.append(pair.index)
# 					print "Appended to non-matches"

# 			except:
# 				if input_value == "stop":
# 					break

# 				else:
# 					possible_match_ids.append(pair.index)
# 					print "Appended to possible matches"

# 		matches = pairs[match_ids].copy()
# 		nonmatches = pairs[nonmatch_ids].copy()
# 		possiblematches = pairs[possible_match_ids].copy()

# 		return matches, possiblematches, nonmatches

# 	def review_connected_components(pairs):

# 		pairs_index_0 = set(pairs.index.levels[pairs.index.names[0]])
# 		pairs_index_1 = set(pairs.index.levels[pairs.index.names[1]])

# 		if set.intersection(pairs_index_0, pairs_index_1) == {}:
# 			raise RuntimeError("Overlapping indexes are not implemented at the moment...")

# 		if self.B:
# 			pairs_graph = nx.Graph()
# 			pairs_graph.add_nodes_from(pairs.index[0], bipartite=0) 
# 			pairs_graph.add_nodes_from(pairs.index[1], bipartite=1) 
# 			pairs_graph.add_edges_from(pairs.index)

# 		else:
# 			pairs_graph = nx.Graph(pairs.index)

# 		print "Stop the review process with the command [stop]"

# 		# Lists with matches, nonmatches and possible matches. 
# 		match_ids = []
# 		nonmatch_ids = []
# 		possible_match_ids = []

# 		for conn in list(nx.connected_components(pairs_graph)) if len(conn) > 1:

# 			if self.B:
# 				records_A, records_B = bipartite.sets(B)
# 			else:
# 				print self.A.ix[conn]

# 			print "Do these records belong to the same entity? [y/n/?]"
# 			sys.stdout.flush()
# 			input_value = raw_input()

# 			try:
# 				same_or_not = distutils.util.strtobool(input_value)

# 				if same_or_not == 1:
# 					match_ids.append(conn.index)
# 					print "Appended to matches"
# 				else:
# 					nonmatch_ids.append(conn.index)
# 					print "Appended to non-matches"

# 			except:
# 				if input_value == "stop":
# 					break

# 				else:
# 					possible_match_ids.append(conn.index)
# 					print "Appended to possible matches"

# 		matches = pairs[match_ids].copy()
# 		nonmatches = pairs[nonmatch_ids].copy()
# 		possiblematches = pairs[possible_match_ids].copy()

# 		return matches, possiblematches, nonmatches




