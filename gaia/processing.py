from .filtering import Filtering
from pandas import DataFrame, merge_asof
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Processing:
    """
    A class for processing and merging data frames.

    Methods
    -------
    run(df1, df2)
        Preprocesses and merges two data frames.
    preprocessing(df)
        Applies preprocessing steps to a data frame.
    remove_nan(df, param='time')
        Removes rows with NaN values in the specified column.
    convert_nan(df, type="mean")
        Replaces NaN values in the data frame based on the specified type.
    normalize_data(df, scaler_type='standard')
        Normalizes the data frame using the specified scaler type.
    interpolation(df, param='linear')
        Interpolates missing values in the data frame using the specified method.
    merge(df1, df2, param='time')
        Merges two data frames on the specified column using an asof merge.
    """

    def __init__(self):
        """
        Initializes the Processing class with an instance of the Filtering class.
        """
        self.filtering = Filtering()

    def run(self, df1, df2):
        """
        Preprocesses and merges two data frames.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first data frame to process and merge.
        df2 : pd.DataFrame
            The second data frame to process and merge.

        Returns
        -------
        pd.DataFrame
            The merged data frame after preprocessing.
        """
        df1 = self.preprocessing(df1)
        df2 = self.preprocessing(df2)
        return self.merge(df1, df2)

    def preprocessing(self, df):
        """
        Applies preprocessing steps to a data frame.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame to preprocess.

        Returns
        -------
        pd.DataFrame
            The preprocessed data frame.
        """
        time = df["time"]
        df = self.remove_nan(df)
        df = self.convert_nan(df)
        df = Filtering.butter_lowpass(df)
        # Uncomment the following lines if interpolation and normalization are needed
        # df = self.interpolation(df)
        # df = self.normalize_data(df, scaler_type='standard')
        # df = self.normalize_data(df, scaler_type='minmax')
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
    
    def convert_nan(self, df, type="mean"):
        """
        Replaces NaN values in the data frame based on the specified type.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame in which to replace NaN values.
        type : str, optional
            The method for replacing NaN values ('mean' or 'zero', default is 'mean').

        Returns
        -------
        pd.DataFrame
            The data frame with NaN values replaced.
        """
        if type == "mean":
            df.fillna(df.mean(), inplace=True)
        elif type == "zero":
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

    def interpolation(self, df, param='linear'):
        """
        Interpolates missing values in the data frame using the specified method.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame in which to interpolate missing values.
        param : str, optional
            The interpolation method to use (default is 'linear').

        Returns
        -------
        pd.DataFrame
            The data frame with interpolated values.
        """
        return df.interpolate(method=param)

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
