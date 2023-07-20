import numpy as np
import pandas


def _compare_exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0):
    # dtypes can be hard if the passed parameters (agreement, disagreement,
    # missing_value) are of different types.
    # http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/03-data-types-and-format.html

    # Convert to pandas.Series if (numpy) arrays are passed.
    if not isinstance(s1, pandas.Series):
        s1 = pandas.Series(s1, index=s1.index)

    if not isinstance(s2, pandas.Series):
        s2 = pandas.Series(s2, index=s2.index)

    # Values or agree/disagree
    if agree_value == "value":
        compare = s1.copy()
        compare[s1 != s2] = disagree_value

    else:
        compare = pandas.Series(disagree_value, index=s1.index)
        compare[s1 == s2] = agree_value

    # Only when disagree value is not identical with the missing value
    if disagree_value != missing_value:
        compare[(s1.isnull() | s2.isnull())] = missing_value

    return compare


def _compare_dates(
    s1, s2, swap_month_day=0.5, swap_months="default", errors="coerce", *args, **kwargs
):
    # validate datatypes
    if str(s1.dtype) != "datetime64[ns]":
        raise ValueError("Left column is not of type datetime64[ns]")

    if str(s2.dtype) != "datetime64[ns]":
        raise ValueError("Right column is not of type datetime64[ns]")

    c = (s1 == s2).astype(np.int64)  # start with int64 (will become float64)

    # The case is which there is a swap_month_day value given.
    if swap_month_day and swap_month_day != 0:
        c[
            (s1.dt.year == s2.dt.year)
            & (s1.dt.month == s2.dt.day)
            & (s1.dt.day == s2.dt.month)
            & (c != 1)
        ] = swap_month_day

    if swap_months and swap_months != 0:
        if swap_months == "default":
            swap_months = [(6, 7, 0.5), (7, 6, 0.5), (9, 10, 0.5), (10, 9, 0.5)]
        else:
            try:
                if not all([len(x) == 3 for x in swap_months]):
                    raise Exception
            except Exception as err:
                raise ValueError(
                    "swap_months must be a list of (first month, \
                    second month, value) tuples or lists. "
                ) from err

        for month1, month2, value in swap_months:
            # if isinstance(value, float):
            #     c = c.astype(np.float64)
            # elif isinstance(value, int):
            #     c = c.astype(np.int64)
            # else:
            #     c = c.astype(object)

            c[
                (s1.dt.year == s2.dt.year)
                & (s1.dt.month == month1)
                & (s2.dt.month == month2)
                & (s1.dt.day == s2.dt.day)
                & (c != 1)
            ] = value

    c = pandas.Series(c)
    c[s1.isnull() | s2.isnull()] = np.nan

    return c
