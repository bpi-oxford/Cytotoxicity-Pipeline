from typing import Any
import pandas as pd
from tqdm import tqdm
import os
import operator
import numpy as np

class CrossTableOperation(object):
    def __init__(self, column, operation, verbose=True):
        """
        Perform cross pandas table operations with given column name
        
        Args:
            column (str): Column name of the inter table operation
            operation (str): Data operation
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "CrossTableOperation"
        self.column = column
        self.operation = operation
        self.verbose = verbose

    def __call__(self, data) -> Any:
        features = data["features"]
        features_out = features[0]

        if self.operation == "divide":
            features_out["{}_{}".format(self.column,self.operation)] = features[0][self.column]/features[1][self.column]

        return {"feature": features_out}
    
class FeatureFilter(object):
    # Mapping of comparison symbols to corresponding functions from the 'operator' module
    OPS = {
        '<': operator.lt,
        '>': operator.gt,
        '==': operator.eq,
        '!=': operator.ne,
        '<=': operator.le,
        '>=': operator.ge,
    }

    def __init__(self, column, operation, value, verbose=True):
        """
        Perform feature filtering in the pandas table with given column name

        Args:
            column (str): Column name to apply the feature filter
            operation (str): Comparison symbols in string
            value (double): Feature filter value
        """
        self.name = "FeatureFilter"
        self.column = column
        self.operation = operation
        self.value = value
        self.verbose = verbose

    def __call__(self, data) -> Any:
        feature = data["feature"]

        # Perform the comparison using the specified operator
        mask = self.OPS[self.operation](feature[self.column], self.value)

        # Apply the mask to filter the DataFrame
        feature_out = feature[mask]

        return {"feature": feature_out}

def intensity_norm(df,channel="mean", suffix="norm",lower=0,upper=1):
    """
    Calculate the normalized cell signal by user given lower and upper bounds and add it as a new column in the dataframe.

    Parameters:
    - df: pandas DataFrame, containing the fluorescence intensity data.
    - channel: str, column name for normalization to take place. (default=mean)
    - suffix: str, suffix appending to the normalized channel output in format of {channel}_{suffix} (default=norm).
    - lower: float, lower bound value for min clipping range (default: 0)
    - upper: float, upper bound value for max clipping range (default: 1)
    
    Returns:
    - df: pandas DataFrame, with a new column '{channel}_{suffix}' containing the normalized signal.
    """
    df["{}_{}".format(channel, suffix)] = df[channel].clip(lower=lower, upper=upper)

    df["{}_{}".format(channel, suffix)] = (df["{}_{}".format(channel, suffix)] - df["{}_{}".format(channel, suffix)].min()) / (df["{}_norm".format(channel)].max() - df["{}_norm".format(channel)].min())

    return df

def intensity_norm_percentile(df,channel="mean", suffix="norm",percentile=1):
    """
    Calculate the normalized cell signal by percentile clippings and add it as a new column in the dataframe.

    Parameters:
    - df: pandas DataFrame, containing the fluorescence intensity data.
    - channel: str, column name for normalization to take place. (default=mean)
    - suffix: str, suffix appending to the normalized channel output in format of {channel}_{suffix} (default=norm).
    - percentile: float, percentile value for min/max range (default: 1)

    Returns:
    - df: pandas DataFrame, with a new column '{channel}_{suffix}' containing the normalized signal.
    - lp: float, the lower percentile value.
    - up: float, the upper percentile value.
    """
    lp = df[channel].quantile(percentile/100.)
    up = df[channel].quantile(1-percentile/100.)

    df = intensity_norm(df, channel=channel, suffix=suffix, lower=lp, upper=up)
    return df, lp, up

def left_table_merge(df1, df2, on=None):
    df = pd.merge(df1, df2, on=on, suffixes=('_left', '_right'))
    columns_to_keep = [col for col in df.columns if not col.endswith('_right')]
    df = df[columns_to_keep]

    rename_dict = {col: col.replace('_left', '') for col in df.columns if col.endswith('_left')}

    # Apply the renaming
    df = df.rename(columns=rename_dict)
    return df

def cal_viability(df,pos_cols=[],neg_cols=[]):
    #TODO: speed up with dask?
    def viability_helper(row):
        exp_pos = 0
        exp_neg = 0

        for pos in pos_cols:
            exp_pos += np.exp(-10*(row[pos]-0.5))

        for neg in neg_cols:
            exp_neg += np.exp(-10*(0.5-row[neg]))

        via = 1/(1+(exp_pos+exp_neg)/(len(pos_cols)+len(neg_cols)))

        return via

    df["viability"] = df.apply(viability_helper, axis=1)
    return df

def calculate_cdi(df, viability_col, death_col, out_col="CDI"):
    """
    Calculate the Cell Death Index (CDI) and add it as a new column in the dataframe.

    Parameters:
    - df: pandas DataFrame containing the fluorescence intensity data.
    - viability_col: str, column name for the viability channel (e.g., live cell marker).
    - death_col: str, column name for the death channel (e.g., dead cell marker).
    - out_col: str, column name for CDI calculation output (default: CDI)

    Returns:
    - df: pandas DataFrame with a new column 'CDI' containing the calculated CDI values.
    """
    # Avoid division by zero by adding a small value to the denominator
    epsilon = 1e-10
    df[out_col] = df[death_col] / (df[viability_col] + df[death_col] + epsilon)
    
    return df