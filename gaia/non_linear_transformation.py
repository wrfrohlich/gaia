import pandas as pd
import numpy as np


class NonLinearTransformation:
    def apply_log_transformation(series):
        """
        Apply log transformation to the data series.

        Parameters
        ----------
        series : pd.Series
            Series of data.

        Returns
        -------
        pd.Series
            Log-transformed data series.
        """
        return np.log1p(series)

    def apply_exponential_transformation(series):
        """
        Apply exponential transformation to the data series.

        Parameters
        ----------
        series : pd.Series
            Series of data.

        Returns
        -------
        pd.Series
            Exponentially-transformed data series.
        """
        return np.exp(series)
