import pandas as pd
import numpy as np

import os

try:
	from faker import Faker
except Exception:
	print 'Faker is not installed. Therefore, the functionalities of this module are limited.'

def addtypos():

	return

def addsubstitutions():

	return

def fakeperson():

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

def dataset(N):

	try:
		fake = Faker()
	except:
		return pd.read_csv('data/censusdataA.csv')

	try:

		for _ in range(0,N):



			print fake.name()

	return None

censusdataA = dataset()