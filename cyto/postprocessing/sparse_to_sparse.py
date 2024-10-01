import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Any
import pandas as pd
from tqdm import tqdm
import os
import operator
import numpy as np
from scipy.optimize import curve_fit

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

def gaussian_kernel(size, sigma):
    """Creates a 1D Gaussian kernel."""
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()

def apply_gaussian_smoothing(values, sigma):
    """Applies Gaussian smoothing using NumPy."""
    kernel_size = int(6 * sigma + 1)  # Common heuristic for kernel size
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_values = np.convolve(values, kernel, mode='same')
    return smoothed_values

def compute_central_difference(values):
    """Computes the gradient using the central finite difference method."""
    gradient = np.zeros_like(values)
    gradient[1:-1] = (values[2:] - values[:-2]) / 2  # Central difference
    gradient[0] = values[1] - values[0]  # Forward difference for the first element
    gradient[-1] = values[-1] - values[-2]  # Backward difference for the last element
    return gradient

def compute_moving_average(data, window_size):
    """Calculates the moving average of a given data array."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def model_func(x, a, b, c):
    """A sample model function for fitting (e.g., a quadratic model)."""
    return a * x**2 + b * x + c  # Change this model based on your needs

def fit_model(x, y):
    """Fit the model to the data using Levenberg-Marquardt and return fitted values."""
    # 2nd order LM fit requires min of 3 points
    if len(x) < 3:
        return y
    
    params, _ = curve_fit(model_func, x, y)
    return model_func(x, *params)

def compute_smoothed_gradient(df, track_id_col='track_id', frame_col='frame', value_col='scalar_value', sigma=2, ma_window=10, smooth_method="gaussian", grad_method="forward"):
    """
    Computes a smoothed time gradient for noisy scalar values in a pandas DataFrame.
    
    Parameters:
    - df: pd.DataFrame containing the data with columns for track ID, frame, and scalar values.
    - track_col: str, name of the column representing track IDs.
    - frame_col: str, name of the column representing frames (time points).
    - value_col: str, name of the column representing scalar values.
    - sigma: float, the standard deviation for Gaussian smoothing (controls the smoothness).
    - ma_window: int, moving average window size
    - smooth_method: str, smooth method between Gaussian smoothing, Levenberg Marquardt method and moving average filter, allowed options: "gaussian", "lm", "ma"
    - grad_method: str, numerical gradient difference method, allow options: "forward", "central"
    
    Returns:
    - pd.DataFrame with additional columns for smoothed scalar values and gradients.
    """
    
    # if isinstance(df, pd.DataFrame):
    #     # Estimate the size of the DataFrame in bytes
    #     df_size = df.memory_usage(deep=True).sum()

    #     # Define a target partition size (e.g., 1 GB)
    #     target_partition_size = 0.1*1024 * 1024 * 1024  # 1GB

    #     # Calculate the number of partitions
    #     npartitions = max(1, int(df_size / target_partition_size))  # At least 1 partition

    #     print("npartitions:", npartitions)

    #     df = dd.from_pandas(df,npartitions)

    # Step 1: Sort the DataFrame by track_id and frame
    df_sorted = df.sort_values(by=[track_id_col, frame_col])

    # Step 2: Interpolate missing frames within each track_id
    df_interpolated = df_sorted.groupby(track_id_col).apply(
        lambda group: group.set_index(frame_col).reindex(
            range(group[frame_col].min(), group[frame_col].max() + 1)
        ).interpolate().reset_index()
    ).reset_index(drop=True)

    # Step 3: Apply Gaussian smoothing to the scalar values
    if smooth_method == "gaussian":
        df_interpolated["{}_smoothed".format(value_col)] = df_interpolated.groupby(track_id_col)[value_col].transform(
            lambda x: apply_gaussian_smoothing(x, sigma=sigma)
        )
    elif smooth_method == "lm":     
        df_interpolated["{}_smoothed".format(value_col)] = df_interpolated.groupby(track_id_col).apply(lambda df: fit_model(df[frame_col].values,df[value_col].values))
    elif smooth_method == "ma":
        df_interpolated["{}_smoothed".format(value_col)] = df_interpolated.groupby(track_id_col)[value_col].transform(
            lambda x: compute_moving_average(x, window_size=ma_window)
        )

    # Step 4: Compute the gradient (rate of change) of the smoothed scalar values
    if grad_method == "forward":
        df_interpolated['{}_grad'.format(value_col)] = df_interpolated.groupby(track_id_col)["{}_smoothed".format(value_col)].transform(np.gradient)
    elif grad_method == "central":
        # central finite difference method
        df_interpolated['{}_grad'.format(value_col)] = df_interpolated.groupby(track_id_col)["{}_smoothed".format(value_col)].transform(
            lambda x: compute_central_difference(x.values)
        )

    # Step 5: Merge the interpolated data back to the original dataframe based on track_id and frame
    result = pd.merge(df, df_interpolated[[track_id_col, frame_col, "{}_smoothed".format(value_col), '{}_grad'.format(value_col)]],
                      on=[track_id_col, frame_col], how='left')

    return result