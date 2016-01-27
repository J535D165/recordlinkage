import pandas
import numpy

from scipy.stats import expon

def generate_vectors(n, p, m, u):

	# Number of matches and non-matches
	n_matches = numpy.floor(p*n)
	n_nonmatches = n - n_matches

	# Create data
	data_matches = (numpy.random.random((n_matches,len(m))) < m).astype(int)
	data_nonmatches = (numpy.random.random((n_nonmatches,len(u))) < u).astype(int)

	# merge data
	data = numpy.append(data_matches, data_nonmatches, axis=0)

	# Make an fake index (index contains only unique record ids.)
	index = numpy.random.choice(numpy.arange(1e5, 1e5+2*n), size=(2,n), replace=False).astype(int)
	
	vectors = pandas.DataFrame(data, index=list(index))

	return vectors.iloc[numpy.random.permutation(len(vectors))], vectors.index[0:n_matches]

# v,i = generate_vectors(1000, 0.05, [0.8]*9, [0.2]*9)
# print(v)
# print (len(i))


