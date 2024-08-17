import numpy as np
import pandas as pd

from gaia.config import Config
from gaia.filtering import Filtering

from pandas import DataFrame, merge_asof
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessing:
    """
    A class for processing and merging data frames.

    Methods
    -------
    run(df1, df2, remove_nan=True, convert_nan='mean', interpolate_method='linear', filter_data=True, normalization=None)
        Preprocesses and merges two data frames with customizable options.
    preprocess(df, remove_nan=True, convert_nan='mean', interpolate_method='linear', filter_data=True, normalization=None)
        Applies preprocessing steps to a data frame with customizable options.
    remove_nan(df, param='time')
        Removes rows with NaN values in the specified column.
    convert_nan(df, method='mean')
        Replaces NaN values in the data frame based on the specified method.
    normalize_data(df, scaler_type='standard')
        Normalizes the data frame using the specified scaler type.
    interpolate(df, method='linear')
        Interpolates missing values in the data frame using the specified method.
    merge(df1, df2, param='time')
        Merges two data frames on the specified column using an asof merge.
    """

    def __init__(self):
        """
        Initializes the Processing class with an instance of the Filtering class.
        """
        self.filtering = Filtering()

    def run(self, df1, df2, df3, remove_nan=True, convert_nan='mean',
            interpolate_method='linear', filter_data="low-pass",
            normalization='standard'):
        """
        Preprocesses and merges two data frames with customizable options.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first data frame to process and merge.
        df2 : pd.DataFrame
            The second data frame to process and merge.
        remove_nan : bool, optional
            Whether to remove rows with NaN values (default is True).
        convert_nan : str, optional
            Method to replace NaN values ('mean', 'zero', or None, default is 'mean').
        interpolate_method : str, optional
            Method for interpolating missing values (default is 'linear').
        filter_data : bool, optional
            Whether to apply a Butterworth low-pass filter (default is True).
        normalization : str, optional
            Method for normalizing the data ('standard' or 'minmax', default is None).

        Returns
        -------
        pd.DataFrame
            The merged data frame after preprocessing.
        """
        df1 = self.preprocess(df1, remove_nan, convert_nan, interpolate_method, filter_data, normalization)
        df2 = self.preprocess(df2, remove_nan, convert_nan, interpolate_method, filter_data, normalization)
        df3 = self.preprocess(df3, remove_nan, convert_nan, interpolate_method, filter_data, normalization)
        merged_data = self.merge(df1, df2)
        merged_data = self.merge(merged_data, df3)
        return merged_data

    def preprocess(self, df, remove_nan=True, convert_nan='mean', interpolate_method='linear', filter_data="low-pass", normalization='standard'):
        """
        Applies preprocessing steps to a data frame with customizable options.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame to preprocess.
        remove_nan : bool, optional
            Whether to remove rows with NaN values (default is True).
        convert_nan : str, optional
            Method to replace NaN values ('mean', 'zero', or None, default is 'mean').
        interpolate_method : str, optional
            Method for interpolating missing values (default is 'linear').
        filter_data : bool, optional
            Whether to apply a Butterworth low-pass filter (default is True).
        normalization : str, optional
            Method for normalizing the data ('standard' or 'minmax', default is None).

        Returns
        -------
        pd.DataFrame
            The preprocessed data frame.
        """
        time = df["time"]
        if time[0]:
            time[0] = 0.00

        if remove_nan:
            df = self.remove_nan(df)
        if interpolate_method:
            df = self.interpolate(df, method=interpolate_method)
        if convert_nan:
            df = self.convert_nan(df, method=convert_nan)
        if filter_data:
            df = Filtering.butter_filter(df, type=filter_data)
        if normalization:
            df = self.normalize_data(df, scaler_type=normalization)

        df["time"] = time
        return df

    def remove_nan(self, df, param='time'):
        """
        Removes rows with NaN values in the specified column.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame from which to remove rows.
        param : str, optional
            The column name to check for NaN values (default is 'time').

        Returns
        -------
        pd.DataFrame
            The data frame with NaN values removed.
        """
        return df.dropna(subset=[param])
    
    def convert_nan(self, df, method="mean"):
        """
        Replaces NaN values in the data frame based on the specified method.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame in which to replace NaN values.
        method : str, optional
            The method for replacing NaN values ('mean' or 'zero', default is 'mean').

        Returns
        -------
        pd.DataFrame
            The data frame with NaN values replaced.
        """
        if method == "mean":
            df.fillna(df.mean(), inplace=True)
        elif method == "zero":
            df.fillna(0, inplace=True)
        return df

    def normalize_data(self, df, scaler_type='standard'):
        """
        Normalizes the data frame using the specified scaler type.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame to normalize.
        scaler_type : str, optional
            The type of scaler to use ('standard' or 'minmax', default is 'standard').

        Returns
        -------
        pd.DataFrame
            The normalized data frame.
        """
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        scaled_data = scaler.fit_transform(df.drop(columns=['time']))
        df_scaled = DataFrame(scaled_data, columns=df.columns.drop('time'))
        df_scaled['time'] = df['time'].values
        return df_scaled

    def interpolate(self, df, method='linear'):
        """
        Interpolates missing values in the data frame using the specified method.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame in which to interpolate missing values.
        method : str, optional
            The interpolation method to use (default is 'linear').

        Returns
        -------
        pd.DataFrame
            The data frame with interpolated values.
        """
        return df.interpolate(method=method)

    def merge(self, df1, df2, param='time'):
        """
        Merges two data frames on the specified column using an asof merge.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first data frame to merge.
        df2 : pd.DataFrame
            The second data frame to merge.
        param : str, optional
            The column name to merge on (default is 'time').

        Returns
        -------
        pd.DataFrame
            The merged data frame.
        """
        df1 = df1.sort_values(param)
        df2 = df2.sort_values(param)

        return merge_asof(df1, df2, on=param)

    def get_magnitude(self, data):
        cfg = Config()

        df = pd.DataFrame()
        for scalar in cfg.scalars:
            df[scalar] = data[scalar]

        for vector in cfg.vectors:
            df[vector] = self.calculate_magnitude(data, vector)
        return df

    def calculate_magnitude(self, data, label):
        """
        Computes the magnitude of vectors for each row in the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data.
        label : str
            The prefix of the columns representing the vector components.

        Returns
        -------
        pd.Series
            A Series containing the magnitudes of the vectors.
        """
        value = data.apply(lambda row: self.create_vector(row, label), axis=1)
        value = value.apply(np.linalg.norm)
        return value

    def create_vector(self, row, prefix):
        """
        Creates a 3D vector from row data.

        Parameters
        ----------
        row : pd.Series
            A row of data from the DataFrame.
        prefix : str
            The prefix of the columns representing the vector components.

        Returns
        -------
        np.array
            A 3D vector.
        """
        return np.array([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]])
