import standartise
import pandas as pd

from functools import wraps


def check_type(func):

	@wraps(func)
	def wrapped(*args, **kwargs):

		result  = func(*args, **kwargs)

		if isinstance(result, (pd.Series, pd.core.series.Series)):
			result = standartise.StandardSeries(result)

		elif isinstance(result, (pd.DataFrame, pd.core.frame.DataFrame)):
			result = standartise.StandardDataFrame(result)

		else:
			pass

		return result

	return wrapped

