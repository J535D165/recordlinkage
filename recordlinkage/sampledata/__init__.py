import pandas as pd
import numpy as np

import os

try:
	from faker import Faker
except Exception:
	print 'Faker is not installed. Therefore, the functionalities of this module are limited.'

personaldata1000A = pd.read_csv('data/personaldata1000A.csv', sep=';')
personaldata1000B = pd.read_csv('data/personaldata1000B.csv', sep=';')

def addtypos(df):

	return

def addsubstitutions(df):

	return

def fakeperson():

	fake = Faker()

	person = {}

	if np.random.random() < 0.5:
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
		df_persons = pd.DataFrame([fakeperson() for _ in range(0, N)])

		# Set an entity id for each created record. 
		df_persons['entity_id'] = np.arange(1, N+1)

		# Reset the index of the dataframe. Start at 1e6
		df_persons.index = np.arange(1e6, 1e6+N)

		# Return the dataframe
		return df_persons

	elif df is not None:

		len_df = N-matches
		if len_df < 0:
			raise ValueError("The number of matches is higher than the number of records N.")

		# Create a dataframe with fake personal information
		df_persons = pd.DataFrame([fakeperson() for _ in range(0, len_df)])

		# Set an entity id for each created record. 
		max_entity_id = max(df['entity_id'])
		df_persons['entity_id'] = np.arange(max_entity_id+1, max_entity_id+len_df+ 1)

		# Dataframe
		df_persons = df.sample(matches).append(pd.DataFrame([fakeperson() for _ in range(0, len_df)]))

		# Reset the index of the dataframe. Start at 1e6
		df_persons.index = np.arange(1e6, 1e6+N)

		# Return the dataframe
		return df_persons

# censusdataA = dataset(1000)
# censusdataB = dataset(1000, censusdataA, 800)
# print censusdataA.head()
# print censusdataB.head()

# censusdataA.to_csv('data/personaldata1000A.csv', sep=';')
# censusdataB.to_csv('data/personaldata1000B.csv', sep=';')



