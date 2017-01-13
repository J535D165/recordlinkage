def merge(df1, df2, matches):
    """merge the data of two dataframes"""

    if len(matches.levels) != 2:
        raise ValueError("the matches need to have 2 levels")

    # get the indices of the matches for both levels
    match_index_1 = matches.get_level_values(0)
    match_index_2 = matches.get_level_values(1)

    raise NotImplementedError()


def update(df1, df2, mask=None):
    """update the data in a dataframe with another dataframe"""
    raise NotImplementedError()


def deduplicate(df1, mask=None):
    """eliminate duplicates of repeating data"""
    raise NotImplementedError()
