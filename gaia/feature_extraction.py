import numpy as np
import pandas as pd

from scipy.fftpack import fft
from scipy.signal import find_peaks

def mean(data):
    return np.mean(data)

def std_dev(data):
    return np.std(data)

def max_value(data):
    return np.max(data)

def min_value(data):
    return np.min(data)

def energy(data):
    return np.sum(np.square(data))

def rms(data):
    return np.sqrt(np.mean(np.square(data)))

def fft_features(data):
    fft_values = fft(data)
    fft_magnitude = np.abs(fft_values)
    return fft_magnitude[:len(fft_magnitude) // 2]

data = pd.DataFrame({
    'time': time,
    'acc_x': acc_x,
    'acc_y': acc_y,
    'acc_z': acc_z,
    'gyro_x': gyro_x,
    'gyro_y': gyro_y,
    'gyro_z': gyro_z,
    'pos_x': position_x,
    'pos_y': position_y,
    'pos_z': position_z
})

features = {
    'mean_acc_x': mean(data['acc_x']),
    'std_acc_x': std_dev(data['acc_x']),
    'max_acc_x': max_value(data['acc_x']),
    'min_acc_x': min_value(data['acc_x']),
    'energy_acc_x': energy(data['acc_x']),
    'rms_acc_x': rms(data['acc_x']),
    'fft_acc_x': fft_features(data['acc_x']),

    'mean_gyro_x': mean(data['gyro_x']),
    'std_gyro_x': std_dev(data['gyro_x']),
    'max_gyro_x': max_value(data['gyro_x']),
    'min_gyro_x': min_value(data['gyro_x']),
    'energy_gyro_x': energy(data['gyro_x']),
    'rms_gyro_x': rms(data['gyro_x']),
    'fft_gyro_x': fft_features(data['gyro_x']),
}

features_df = pd.DataFrame([features])
print(features_df)

def velocity(position, time):
    return np.gradient(position, time)

def acceleration(velocity, time):
    return np.gradient(velocity, time)

def distance_traveled(position):
    return np.sum(np.sqrt(np.diff(position)**2))

def trajectory_smoothness(position):
    velocity = np.gradient(position)
    acceleration = np.gradient(velocity)
    jerk = np.gradient(acceleration)
    return np.sum(np.abs(jerk))

pos_x_velocity = velocity(data['pos_x'], data['time'])
pos_x_acceleration = acceleration(pos_x_velocity, data['time'])

camera_features = {
    'mean_pos_x': mean(data['pos_x']),
    'std_pos_x': std_dev(data['pos_x']),
    'max_pos_x': max_value(data['pos_x']),
    'min_pos_x': min_value(data['pos_x']),
    'distance_pos_x': distance_traveled(data['pos_x']),
    'smoothness_pos_x': trajectory_smoothness(data['pos_x']),
}

camera_features_df = pd.DataFrame([camera_features])
print(camera_features_df)
