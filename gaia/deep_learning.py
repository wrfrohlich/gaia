import logging
import pandas as pd
from os import path, makedirs
from sklearn.model_selection import train_test_split

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

