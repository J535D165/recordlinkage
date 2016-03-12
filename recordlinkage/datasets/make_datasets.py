import pandas
import numpy

import os


def _krebsregister_block(block):

	fp_i = os.path.join(os.path.dirname(__file__), 'data', 'krebsregister', 'block_{}.csv'.format(block))

	data_block = pandas.read_csv(fp_i, index_col=['id_1', 'id_2'], na_values='?')
	data_block.columns = ['cmp_firstname1', 'cmp_firstname2', 'cmp_lastname1', 'cmp_lastname2', 'cmp_sex', 'cmp_birthday', 'cmp_birthmonth', 'cmp_birthyear', 'cmp_zipcode', 'is_match']
	data_block.index.names = ['id1', 'id2']
	return data_block.sample(n=len(data_block), replace=False, random_state=100+block)
