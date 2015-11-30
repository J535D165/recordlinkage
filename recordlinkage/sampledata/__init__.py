# __init__.py

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright (C) 2015 Jonathan de Bruin

import pandas as pd
import numpy as np

import os

try:
	from faker import Faker
except Exception:
	print 'Faker is not installed. Therefore, the functionalities of this module are limited.'

personaldata1000A = pd.read_csv('recordlinkage/sampledata/data/personaldata1000A.csv', sep=';', encoding='utf-8')
personaldata1000B = pd.read_csv('recordlinkage/sampledata/data/personaldata1000B.csv', sep=';', encoding='utf-8')

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
			to_replace = np.random.choice([True, False], len(df), p=[missing_dict[col], 1-missing_dict[col]])
			df.loc[to_replace, col] = np.nan

	return df

def addsubstitutions(df, subs_dict=SUBS_DICT):

	fake = Faker()

	for col in list(df):

		if col in subs_dict.keys():

			# Make a random sample of values to replace. 
			to_replace = np.random.choice([True, False], len(df), p=[subs_dict[col], 1-subs_dict[col]])

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

# censusdataA.to_csv('data/personaldata1000A.csv', sep=';')
# censusdataB.to_csv('data/personaldata1000B.csv', sep=';')



