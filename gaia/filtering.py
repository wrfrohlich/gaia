import numpy as np
from scipy.signal import butter, filtfilt

class Filtering:
    @staticmethod
    def butter_lowpass_filter(data, cutoff=3.0, fs=100, order=5):
        if cutoff <= 0 or fs <= 0:
            raise ValueError("Cutoff frequency and sampling frequency must be greater than zero.")

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            raise ValueError("Normalized cutoff frequency must be between 0 and 1.")

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_lowpass(data):
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = Filtering.butter_lowpass_filter(data[col].values)
        return data

