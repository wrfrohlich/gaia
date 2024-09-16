import logging
import pandas as pd
import matplotlib.pyplot as plt

from os import path, makedirs
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from gaia.config import Config
class DeepLearning:
    def __init__(self, name):
        config = Config()
        self.path_experiment = path.join(config.experiments, name) if name else config.experiments
        self.points = config.body_parts

        # Create experiment directory if it doesn't exist
        if not path.exists(self.path_experiment):
            makedirs(self.path_experiment)
        
        logging.basicConfig(level=logging.INFO)

    def sep_data(self):
        correlation_report = pd.read_csv(f'{self.path_experiment}/cross_correlation_report.csv', sep=";")
        correlation_data = pd.read_csv(f'{self.path_experiment}/cross_correlation_results.csv')
        data = {}

        for _, row in correlation_report.iterrows():
            cleaned_list = [col for col in row if pd.notnull(col)]
            data[cleaned_list[0]] = {}
            for imu in cleaned_list[1:]:
                lag = correlation_data[(correlation_data["kinematic"] == cleaned_list[0]) & (correlation_data["imu"] == imu)].lag.item()
                data[cleaned_list[0]][imu] = lag

        return data

    def prep_data(self, item, sep_data, data):
        x = data[sep_data[item].keys()].copy()
        y = data[item].copy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def run(self, sep_data, data):
        for item in sep_data:
            x_train, x_test, y_train, y_test = self.prep_data(item, sep_data, data)

    def run(self, sep_data, data, method="linear_regression"):
        for item in sep_data:
            x_train, x_test, y_train, y_test = self.prep_data(item, sep_data, data)
            model = Sequential()

            model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(3))  # Saída com 3 neurônios (correspondente a 'c7_x', 'c7_y', 'c7_z')

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            history = model.fit(x_train, y_train, validation_data=(x_train, y_test), epochs=50, batch_size=32)

            loss, mae = model.evaluate(x_train, y_test)
            print(f'MAE: {mae}')

            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()



