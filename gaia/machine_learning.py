import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import path, makedirs
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

from gaia.config import Config


class MachineLearning:
    def __init__(self, name):
        config = Config()
        self.path_experiment = (
            path.join(config.experiments, name) if name else config.experiments
        )
        self.points = config.body_parts

        # Create experiment directory if it doesn't exist
        if not path.exists(self.path_experiment):
            makedirs(self.path_experiment)

        logging.basicConfig(level=logging.INFO)

    def sep_data(self):
        correlation_report = pd.read_csv(
            f"{self.path_experiment}/cross_correlation_report.csv", sep=";"
        )
        correlation_data = pd.read_csv(
            f"{self.path_experiment}/cross_correlation_results.csv"
        )
        data = {}

        for _, row in correlation_report.iterrows():
            cleaned_list = [col for col in row if pd.notnull(col)]
            data[cleaned_list[0]] = {}
            for imu in cleaned_list[1:]:
                lag = correlation_data[
                    (correlation_data["kinematic"] == cleaned_list[0])
                    & (correlation_data["imu"] == imu)
                ].lag.item()
                data[cleaned_list[0]][imu] = lag

        return data

    def prep_data(self, item, sep_data, data):
        x = data[sep_data[item].keys()].copy()
        y = data[item].copy()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    def run(
        self,
        sep_data,
        data,
        method: Literal["linear_regression", "random_forest"] = "linear_regression",
    ):
        for item in sep_data:
            x_train, x_test, y_train, y_test = self.prep_data(item, sep_data, data)
            model = self.get_method(method=method)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            if method == "random_forest":
                self.plot_feature_importance(item, model, x_train)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rae = sum(abs(y_test - y_pred)) / sum(abs(y_test - y_test.mean()))
            mape = mean_absolute_percentage_error(y_test, y_pred)

            report = pd.DataFrame(
                {"MSE": [mse], "R2": [r2], "MAE": [mae], "RAE": [rae], "MAPE": [mape]}
            )

            report.insert(0, "item", item)
            report.to_csv(
                f"{self.path_experiment}/report_{method}.csv",
                mode="a",
                header=not path.exists(f"{self.path_experiment}/report_{method}.csv"),
                index=False,
            )

    def plot_feature_importance(self, item, model, x_train, fontsize=15):
        importance = model.feature_importances_
        features = x_train.columns

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importance, y=features)

        plt.xlabel("Importance (%)", fontsize=fontsize)
        plt.ylabel("IMU", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()

        plt.savefig(f"{self.path_experiment}/feature_importance_{item}.png")
        plt.clf()
        plt.close()

        # Criar o relatório em CSV
        report = pd.DataFrame({"feature": features, "importance": importance})

        report.insert(0, "item", item)
        report.to_csv(
            f"{self.path_experiment}/feature_importance.csv",
            mode="a",
            header=not path.exists(f"{self.path_experiment}/feature_importance.csv"),
            index=False,
        )

    def tune_hyperparameters(self, method):
        param_grid = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
        grid_search = GridSearchCV(
            RandomForestRegressor(), param_grid, cv=3, scoring="r2"
        )
        return grid_search

    def get_method(self, method="linear_regression"):
        model = None
        if method == "linear_regression":
            model = LinearRegression()
        elif method == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == "xgboost":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        elif method == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            )
        return model
