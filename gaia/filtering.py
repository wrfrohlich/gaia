import numpy as np
from scipy.signal import butter, filtfilt


class Filtering:
    """
    A class for applying Butterworth filters to data.

    Methods
    -------
    butter_lowpass_filter(data, cutoff=3.0, fs=100, order=5)
        Applies a Butterworth low-pass filter to the given data.
    butter_lowpass(data)
        Applies the Butterworth low-pass filter to all numeric columns in a DataFrame.
    butter_highpass_filter(data, cutoff=0.5, fs=100, order=5)
        Applies a Butterworth high-pass filter to the given data.
    butter_bandpass_filter(data, lowcut=0.5, highcut=3.0, fs=100, order=5)
        Applies a Butterworth band-pass filter to the given data.
    """

    @staticmethod
    def butter_filter(data, type="low-pass"):
        """
        Apply the Butterworth low-pass filter to all numeric columns in a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data to filter.

        Returns
        -------
        pd.DataFrame
            The DataFrame with filtered numeric columns.
        """
        for col in data.select_dtypes(include=[np.number]).columns:
            if type == "low-pass":
                data[col] = Filtering.butter_lowpass_filter(data[col].values)
            if type == "high-pass":
                data[col] = Filtering.butter_highpass_filter(data[col].values)
            if type == "band-pass":
                data[col] = Filtering.butter_bandpass_filter(data[col].values)
        return data

    @staticmethod
    def butter_lowpass_filter(data, cutoff=3.0, fs=100, order=5):
        """
        Apply a Butterworth low-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to filter.
        cutoff : float, optional
            The cutoff frequency of the filter (default is 3.0 Hz).
        fs : int, optional
            The sampling frequency of the data (default is 100 Hz).
        order : int, optional
            The order of the filter (default is 5).

        Returns
        -------
        np.ndarray
            The filtered data.

        Raises
        ------
        ValueError
            If cutoff frequency or sampling frequency is non-positive or if normalized cutoff frequency is out of range.
        """
        if cutoff <= 0 or fs <= 0:
            raise ValueError(
                "Cutoff frequency and sampling frequency must be greater than zero."
            )

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            raise ValueError("Normalized cutoff frequency must be between 0 and 1.")

        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_highpass_filter(data, cutoff=0.5, fs=100, order=5):
        """
        Apply a Butterworth high-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to filter.
        cutoff : float, optional
            The cutoff frequency of the filter (default is 0.5 Hz).
        fs : int, optional
            The sampling frequency of the data (default is 100 Hz).
        order : int, optional
            The order of the filter (default is 5).

        Returns
        -------
        np.ndarray
            The filtered data.

        Raises
        ------
        ValueError
            If cutoff frequency or sampling frequency is non-positive or if normalized cutoff frequency is out of range.
        """
        if cutoff <= 0 or fs <= 0:
            raise ValueError(
                "Cutoff frequency and sampling frequency must be greater than zero."
            )

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            raise ValueError("Normalized cutoff frequency must be between 0 and 1.")

        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_bandpass_filter(data, lowcut=0.5, highcut=3.0, fs=100, order=5):
        """
        Apply a Butterworth band-pass filter to the given data.

        Parameters
        ----------
        data : np.ndarray
            The data to filter.
        lowcut : float, optional
            The lower cutoff frequency of the filter (default is 0.5 Hz).
        highcut : float, optional
            The higher cutoff frequency of the filter (default is 3.0 Hz).
        fs : int, optional
            The sampling frequency of the data (default is 100 Hz).
        order : int, optional
            The order of the filter (default is 5).

        Returns
        -------
        np.ndarray
            The filtered data.

        Raises
        ------
        ValueError
            If cutoff frequencies or sampling frequency are non-positive or if normalized cutoff frequencies are out of range.
        """
        if lowcut <= 0 or highcut <= 0 or fs <= 0:
            raise ValueError(
                "Cutoff frequencies and sampling frequency must be greater than zero."
            )

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        if low <= 0 or high <= 0 or low >= 1 or high >= 1:
            raise ValueError("Normalized cutoff frequencies must be between 0 and 1.")

        b, a = butter(order, [low, high], btype="band", analog=False)
        y = filtfilt(b, a, data)
        return y
