from __future__ import division

import random 

import pandas
import numpy

M_DEFAULT = [
	(0.5, 0.5), (0.8, 0.9), (0.85, 0.9), 
	(0.9, 0.95), (0.6, 0.9), (0.85, 0.95),
	(0.85, 0.95), (0.9, 0.9), (0.5, 0.5),
	(0.6, 0.9), (0.95, 1.0), (0.89, 0.92),
	(0.85, 1.0), (0.85, 0.85), (0.84, 0.93),
	(0.7, 0.9), (0.85, 1.0), (0.9, 0.9),
	(0.5, 0.5), (0.6, 0.6)
	]

U_DEFAULT = [
	(0.5, 0.5), (0.1, 0.4), (0.15, 0.2), 
	(0.0, 0.5), (0.05, 0.09), (0.15, 0.25),
	(0.25, 0.25), (0.1, 0.3), (0.1, 0.3),
	(0.1, 0.3), (0.15, 0.2), (0.2, 0.5),
	(0.1, 0.1), (0.02, 0.02), (0.02, 0.3),
	(0.2, 0.5), (0.05, 0.35), (0.0, 0.4)
	]

def simulate_similarity(N, x0, x1):
	"""
	Linear similarity
	f(x) = f0 + a*x
	
	two equations
	1 = f0 + a*x0
	0 = f0 + a*x1
	
	solve
	1 - a*x0 = -a*x1
	1 = -a*x1 + a*x0 = a*(x0-x1)
	a = 1/(x0-x1)
	
	f0 = -x1/(x0-x1) = -x1*a

	"""

	try:

		if x0 > x1:
			raise ValueError('x1 needs to be larger than or equal to x0')

		a = 1/(x0-x1)
		f0 = -x1*a

		fx = f0 + a*numpy.random.random(N) 
		
		fx[(fx > 1)] = 1
		fx[(fx < 0)] = 0

		return fx

	except (ZeroDivisionError, TypeError):

		return (numpy.random.random(N) < x0).astype(int)

def simulate_features(n=10000, p=0.2, m=None, u=None, n_features=8):

	# Number of matches and non-matches
	n_matches = numpy.floor(p*n)
	n_nonmatches = n - n_matches

	if not m and not u:
		m = random.sample(M_DEFAULT, n_features)
		u = random.sample(U_DEFAULT, n_features)

	# Repair single values 
	for i in range(0,len(m)):
		if not isinstance(m[i],(list, dict, tuple)):
			m[i] = (m[i], m[i])

	for i in range(0,len(u)):
		if not isinstance(u[i],(list, dict, tuple)):
			u[i] = (u[i], u[i])

	# Create data
	data_matches = numpy.array([simulate_similarity(n_matches, mi[0], mi[1]) for mi in list(m)]).T
	data_nonmatches = numpy.array([simulate_similarity(n_nonmatches, ui[0], ui[1]) for ui in list(u)]).T

	# merge data
	data = numpy.append(data_matches, data_nonmatches, axis=0)

	# Make an fake index (index contains only unique record ids.)
	index = numpy.random.choice(numpy.arange(1e5, 1e5+2*n), size=(2,n), replace=False).astype(int)
	
	vectors = pandas.DataFrame(data, index=list(index))

	return vectors.iloc[numpy.random.permutation(len(vectors))], vectors.index[0:n_matches]

# features, m = simulate_features(1000000, 0.05, [[0.5, 0.9]]*9, [[0.1, 0.5]]*9)
# print(features)
# print (len(m))
# f,i = simulate_features()
# print(f)
# print len(i)


