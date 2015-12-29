import standardise
import pandas as pd

from functools import wraps


def check_type(func):     
	"""     
	Check the type of the returned value. If the StandardSeries or
	StandardDataFrame needs to be returned, check if this is indeed the correct type. If not, convert it
	into the correct type.      
	"""

	@wraps(func)
	def wrapped(*args, **kwargs):

		result  = func(*args, **kwargs)

		if isinstance(result, (pd.Series, pd.core.series.Series)):
			result = standardise.StandardSeries(result)

		elif isinstance(result, (pd.DataFrame, pd.core.frame.DataFrame)):
			result = standardise.StandardDataFrame(result)

		else:
			return

		return result

	return wrapped

def _check_index_uniqueness(dataframe_or_series):

	if not dataframe_or_series.index.is_unique:
		raise ValueError("The index values of %s '%s' are not unique. This is required.", (type(dataframe_or_series), dataframe_or_series))
