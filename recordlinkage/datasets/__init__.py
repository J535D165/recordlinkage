
import pandas
import numpy

import os

try:
	from faker import Faker
except ImportError:
	print ('Faker is not installed. Therefore, the functionalities of this module are limited.')

def krebsregister_cmp_data(*args, **kwargs):

	"""
	krebsregister_cmp_data(block=1)

	This dataset of comparison patterns was obtained in a epidemiological
	cancer study in Germany. The comparison patterns were created by the
	Institute for Medical Biostatistics, Epidemiology and Informatics (IMBEI)
	and the University Medical Center of Johannes Gutenberg University
	(Mainz, Germany). The dataset is available for research online.

		*"The records represent individual data including first and  family
		name, sex, date of birth and postal code, which were collected through
		iterative insertions in the course of several years. The comparison
		patterns in this data set are based on a sample of 100.000 records
		dating from 2005 to 2008. Data pairs were classified as "match" or
		"non-match" during  an extensive manual review where several
		documentarists were involved.  The resulting classification formed the
		basis for assessing the quality of the registry's own record linkage
		procedure.*

		*In order to limit the amount of patterns a blocking procedure was
		applied, which selects only record pairs that meet specific agreement
		conditions. The results of the following six blocking iterations were
		merged together:* 

		1. *Phonetic equality of first name and family name, equality of date of birth.*
		2. *Phonetic equality of first name, equality of day of birth.*
		3. *Phonetic equality of first name, equality of month of birth.*
		4. *Phonetic equality of first name, equality of year of birth.*
		5. *Equality of complete date of birth.*
		6. *Phonetic equality of family name, equality of sex.*

		*This procedure resulted in 5.749.132 record pairs, of which 20.931 are
		matches. The data set is split into 10 blocks of (approximately) equal
		size and ratio of matches to non-matches."*

	:param block: An integer or a list with integers between 1 and 10. The blocks are the blocks explained in the description.

	:return: A data frame with comparison vectors and a multi index with the indices of the matches.  
	:rtype: (pandas.DataFrame, pandas.MultiIndex)

	"""

	try:
		from rldatasets import krebsregister_cmp_data
		return krebsregister_cmp_data(*args, **kwargs)

	except ImportError:
		print("Install recordlinkage-datasets to use this dataset.")

def load_censusA():

	fp = os.path.join(os.path.dirname(__file__), 'data', 'personaldata1000A.csv')

	df = pandas.read_csv(fp, sep=';', index_col='record_id', encoding='utf-8')
	df.index.name = 'index_A'
	return df

def load_censusB():

	fp = os.path.join(os.path.dirname(__file__), 'data', 'personaldata1000B.csv')

	df = pandas.read_csv(fp, sep=';', index_col='record_id', encoding='utf-8')
	df.index.name = 'index_B'
	return df

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

# censusdataA = dataset(1000)
# censusdataB = dataset(1000, censusdataA, 800)
# print censusdataA.head()
# print censusdataB.head()
# print censusdataB.dtypes

# censusdataA.to_csv('data/personaldata1000A.csv', sep=';')
# censusdataB.to_csv('data/personaldata1000B.csv', sep=';')



