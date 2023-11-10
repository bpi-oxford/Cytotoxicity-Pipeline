from typing import Any
import pandas as pd
from tqdm import tqdm
import os

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