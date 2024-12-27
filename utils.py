import pandas as pd
import numpy as np


def find_movies(dataframe: pd.DataFrame, column_name: str, query_string: str):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe")

    mask = dataframe[column_name].str.contains(query_string, case=False, na=False)
    matching_movies = dataframe[mask]
    result = matching_movies[[column_name]].reset_index()
    return result
