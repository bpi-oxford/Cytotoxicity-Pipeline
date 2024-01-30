from typing import Any
import pandas as pd
from tqdm import tqdm
import os
import operator

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