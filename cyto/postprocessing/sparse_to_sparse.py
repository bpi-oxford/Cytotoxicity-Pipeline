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

def intensity_norm(df,channel="mean",percentile=1):
    lp = df[channel].quantile(percentile/100.)
    up = df[channel].quantile(1-percentile/100.)

    df["{}_norm".format(channel)] = df[channel].clip(lower=lp, upper=up)

    df["{}_norm".format(channel)] = (df["{}_norm".format(channel)] - df["{}_norm".format(channel)].min()) / (df["{}_norm".format(channel)].max() - df["{}_norm".format(channel)].min())

    return df

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