import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt

# List of important points in the gold standard data
important_points = [
    'sacrum_x', 'sacrum_y', 'sacrum_z',
    'r asis_x', 'r asis_y', 'r asis_z', 'l asis_x', 'l asis_y', 'l asis_z',
    'r knee 1_x', 'r knee 1_y', 'r knee 1_z', 'l knee 1_x', 'l knee 1_y', 'l knee 1_z',
    'r mall_x', 'r mall_y', 'r mall_z', 'l mall_x', 'l mall_y', 'l mall_z',
    'r heel_x', 'r heel_y', 'r heel_z', 'l heel_x', 'l heel_y', 'l heel_z',
    'r met_x', 'r met_y', 'r met_z', 'l met_x', 'l met_y', 'l met_z'
]

def build_model(lstm_units=64, dropout_rate=0.2, optimizer='adam'):
    """
    Build and compile a Keras LSTM model.
    
    Parameters:
        lstm_units (int): Number of units in LSTM layers.
        dropout_rate (float): Dropout rate for Dropout layers.
        optimizer (str): Optimizer for model compilation.
    
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(1, 9)))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(important_points)))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def preprocess_data(df_imu, df_gold):
    """
    Preprocess the IMU and gold standard data.
    
    Parameters:
        df_imu (DataFrame): IMU data.
        df_gold (DataFrame): Gold standard data.
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test sets for model training.
    """
    # Synchronize data by time
    df_combined = pd.merge_asof(df_imu, df_gold, on='time', direction='nearest')

    # Separate inputs (IMU) and outputs (gold standard)
    X = df_combined[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw']]
    y = df_combined[important_points]

    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Reshape for LSTM input [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler_y

def perform_grid_search(X_train, y_train):
    """
    Perform Grid Search to find the best hyperparameters.
    
    Parameters:
        X_train (ndarray): Training input data.
        y_train (ndarray): Training output data.
    
    Returns:
        best_model: Trained Keras model with the best hyperparameters.
        best_params (dict): Best hyperparameters found by Grid Search.
    """
    # Create KerasRegressor model
    model = KerasRegressor(model=build_model, verbose=0)

    # Define parameter grid
    param_grid = {
        'model__lstm_units': [50, 64, 100],
        'model__dropout_rate': [0.2, 0.3],
        'model__optimizer': ['adam', 'rmsprop'],
        'epochs': [50, 100],
        'batch_size': [32, 64]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    best_params = grid_result.best_params_
    print(f"Best parameters: {best_params}")

    return grid_result.best_estimator_, best_params

def train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, scaler_y):
    """
    Train and evaluate the Keras model with the best parameters.
    
    Parameters:
        X_train (ndarray): Training input data.
        y_train (ndarray): Training output data.
        X_test (ndarray): Test input data.
        y_test (ndarray): Test output data.
        best_params (dict): Best hyperparameters for model training.
        scaler_y (StandardScaler): Scaler used to inverse transform the outputs.
    
    Returns:
        None
    """
    # Build and train the model with best parameters
    best_model = build_model(
        optimizer=best_params['model__optimizer'],
        dropout_rate=best_params['model__dropout_rate'],
        lstm_units=best_params['model__lstm_units']
    )
    history = best_model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_test, y_test)
    )

    # Evaluate the model
    loss = best_model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Inverse transform predictions and test data
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Compare predictions with gold standard
    for i in range(len(y_test_inverse)):
        print(f"Real: {y_test_inverse[i]}, Predicted: {y_pred_inverse[i]}")

    # Plot predictions vs. gold standard
    for i in range(len(important_points)):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inverse[:, i], label='Real')
        plt.plot(y_pred_inverse[:, i], label='Predicted')
        plt.title(f'Comparison: {important_points[i]}')
        plt.xlabel('Samples')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'deep_learning_{important_points[i]}.png')
        plt.close()

def deep_learning(df_imu, df_gold):
    """
    Execute the deep learning pipeline for IMU data and gold standard data.
    
    Parameters:
        df_imu (DataFrame): IMU data.
        df_gold (DataFrame): Gold standard data.
    
    Returns:
        None
    """
    X_train, X_test, y_train, y_test, scaler_y = preprocess_data(df_imu, df_gold)
    best_model, best_params = perform_grid_search(X_train, y_train)
    train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, scaler_y)
