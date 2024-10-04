import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import path, makedirs
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from gaia.config import Config


class DeepLearning:
    def __init__(self, name):
        config = Config()
        self.path_experiment = (
            path.join(config.experiments, name) if name else config.experiments
        )
        self.points = config.body_parts

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

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=(input_shape,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def run(self, sep_data, data):
        for item in sep_data:
            x_train, x_test, y_train, y_test = self.prep_data(item, sep_data, data)
            model = self.build_model(input_shape=x_train.shape[1])

            early_stop = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
            )

            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            report = pd.DataFrame(
                {"MSE": [mse], "R2": [r2], "MAE": [mae], "MAPE": [mape]}
            )
            report.insert(0, "item", item)
            report.to_csv(
                f"{self.path_experiment}/report_deep_learning.csv",
                mode="a",
                header=not path.exists(
                    f"{self.path_experiment}/report_deep_learning.csv"
                ),
                index=False,
            )

    def plot_training_history(self, history, item):
        plt.figure(figsize=(10, 8))
        plt.plot(history.history["loss"], label="Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.title(f"Training History for {item}", fontsize=15)
        plt.legend()
        plt.savefig(f"{self.path_experiment}/training_history_{item}.png")
        plt.clf()
        plt.close()
