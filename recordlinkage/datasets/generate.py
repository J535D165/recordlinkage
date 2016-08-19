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


MISSING_DICT  = {
	'first_name': 0.02,
	'sex': 0.02,
	'last_name': 0.02,
	'phone_number': 0.1,
	'job': 0.15,
	'email': 0.1,
	'birthdate': 0.005,
	'street_address': 0.08,
	'postcode': 0.08,
	'city': 0.01
}

SUBS_DICT  = {
	'first_name': 0.02,
	'sex': 0.002,
	'last_name': 0.02,
	'phone_number': 0.1,
	'job': 0.1,
	'email': 0.08,
	'birthdate': 0.005,
	'street_address': 0.05,
	'postcode': 0.08,
	'city': 0.01
}

def addtypos(df):

	return

def addmissingvalues(df, missing_dict=MISSING_DICT):

	fake = Faker()

	for col in list(df):

		if col in missing_dict.keys():

			# Make a random sample of values to replace. 
			to_replace = numpy.random.choice([True, False], len(df), p=[missing_dict[col], 1-missing_dict[col]])
			df.loc[to_replace, col] = numpy.nan

	return df

def addsubstitutions(df, subs_dict=SUBS_DICT):

	fake = Faker()

	for col in list(df):

		if col in subs_dict.keys():

			# Make a random sample of values to replace. 
			to_replace = numpy.random.choice([True, False], len(df), p=[subs_dict[col], 1-subs_dict[col]])

			# Special case when for first name
			if col == 'first_name':

				if 'sex' == 'M':
					df.loc[ to_replace, col] = [fake.first_name_male() for _ in range(0,sum(to_replace))]
				elif 'sex' == 'F':
					df.loc[ to_replace, col] = [fake.first_name_female() for _ in range(0,sum(to_replace))]
				else:
					pass
			elif col == 'last_name':
				df.loc[ to_replace, col] = [fake.last_name() for _ in range(0,sum(to_replace))]
			elif col == 'phone_number':
				df.loc[ to_replace, col] = [fake.phone_number() for _ in range(0,sum(to_replace))]
			elif col == 'job':
				df.loc[ to_replace, col] = [fake.job() for _ in range(0,sum(to_replace))]
			elif col == 'email':
				df.loc[ to_replace, col] = [fake.free_email() for _ in range(0,sum(to_replace))]
			elif col == 'birthdate':
				df.loc[ to_replace, col] = [fake.date() for _ in range(0,sum(to_replace))]
			elif col == 'street_address':
				df.loc[ to_replace, col] = [fake.street_address() for _ in range(0,sum(to_replace))]
			elif col == 'postcode':
				df.loc[ to_replace, col] = [fake.postcode() for _ in range(0,sum(to_replace))]
			elif col == 'city':
				df.loc[ to_replace, col] = [fake.city() for _ in range(0,sum(to_replace))]

	return df

def fakeperson():

	fake = Faker()

	person = {}

	if numpy.random.random() < 0.5:
		person['first_name'] = fake.first_name_male()
		person['sex'] = 'M'
	else:
		person['first_name'] = fake.first_name_female()
		person['sex'] = 'F'

	person['last_name'] = fake.last_name()
	person['phone_number'] = fake.phone_number()
	person['job'] = fake.job()
	person['email'] = fake.free_email()
	person['birthdate'] = fake.date()
	person['street_address'] = fake.street_address()
	person['postcode'] = fake.postcode()
	person['city'] = fake.city()

	return person

def dataset(N, df=None, matches=None):

	if df is None:

		# Create a dataframe with fake personal information
		df_persons = pandas.DataFrame([fakeperson() for _ in range(0, N)])

		# Set an entity id for each created record. 
		df_persons['entity_id'] = numpy.arange(1, N+1).astype(object)

		# Reset the index of the dataframe. Start at 1e6
		df_persons.set_index(numpy.arange(1e6, 1e6+N).astype(numpy.int64), inplace=True)
		df_persons.index.name = 'record_id'

		# Return the dataframe
		return df_persons

	elif df is not None:

		len_df = N-matches
		if len_df < 0:
			raise ValueError("The number of matches is higher than the number of records N.")

		# Create a dataframe with fake personal information
		df_persons = pandas.DataFrame([fakeperson() for _ in range(0, len_df)])

		# Set an entity id for each created record. 
		max_entity_id = max(df['entity_id'])
		df_persons['entity_id'] = numpy.arange(max_entity_id+1, max_entity_id+len_df+ 1).astype(object)

		# Dataframe
		df_persons = df.sample(matches).append(pandas.DataFrame([fakeperson() for _ in range(0, len_df)]))

		# Reset the index of the dataframe. Start at 1e6
		df_persons.set_index(numpy.arange(1e6, 1e6+N).astype(numpy.int64), inplace=True)
		df_persons.index.name = 'record_id'

		# Add substituations.
		df_persons = addsubstitutions(df_persons)

		# Add missing values
		df_persons = addmissingvalues(df_persons)

		# Return the dataframe
		return df_persons
