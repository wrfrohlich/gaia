import logging
import pandas as pd
from os import path, makedirs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from gaia.config import Config

class MachineLearning:
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

    def run(self, sep_data, data, method="linear_regression"):
        for item in sep_data:
            x_train, x_test, y_train, y_test = self.prep_data(item, sep_data, data)
            model = self.get_method(method=method)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rae = sum(abs(y_test - y_pred)) / sum(abs(y_test - y_test.mean()))

            print(f"{item} -> MSE: {mse} / R^2: {r2} - MAE: {mae} - RAE: {rae}")

    def get_method(self, method="linear_regression"):
        model = None
        if method == "linear_regression":
            model = LinearRegression()
        elif method == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == "--":
            pass
        elif method == "---":
            pass
        return model

