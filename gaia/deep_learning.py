import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt

important_points = [
    'sacrum_x', 'sacrum_y', 'sacrum_z',
    'r asis_x', 'r asis_y', 'r asis_z', 'l asis_x', 'l asis_y', 'l asis_z',
    'r knee 1_x', 'r knee 1_y', 'r knee 1_z', 'l knee 1_x', 'l knee 1_y', 'l knee 1_z',
    'r mall_x', 'r mall_y', 'r mall_z', 'l mall_x', 'l mall_y', 'l mall_z',
    'r heel_x', 'r heel_y', 'r heel_z', 'l heel_x', 'l heel_y', 'l heel_z',
    'r met_x', 'r met_y', 'r met_z', 'l met_x', 'l met_y', 'l met_z'
]

def build_model(lstm_units=64, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(1, 9), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(important_points)))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def deep_learning(df_imu, df_gold):
    # Sincronizar os dados pelo tempo
    df_combined = pd.merge_asof(df_imu, df_gold, on='time', direction='nearest')

    # Separar as entradas (IMU) e saídas (padrão ouro)
    X = df_combined[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw']]
    y = df_combined[important_points]

    # Normalizar os dados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Reshape para entrada do LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Construir o modelo usando KerasRegressor para validação cruzada e ajuste de hiperparâmetros
    model = KerasRegressor(model=build_model, verbose=0)

    # Definir os parâmetros para busca em grade
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
    print(f"Melhores parâmetros: {best_params}")

    # Treinar o modelo com os melhores parâmetros
    best_model = build_model(optimizer=best_params['model__optimizer'], dropout_rate=best_params['model__dropout_rate'], lstm_units=best_params['model__lstm_units'])
    history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_test, y_test))

    # Avaliar o modelo
    loss = best_model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')

    # Previsões
    y_pred = best_model.predict(X_test)

    # Inverter a normalização das previsões para comparação
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Comparar previsões com o padrão ouro
    for i in range(len(y_test_inverse)):
        print(f"Real: {y_test_inverse[i]}, Predito: {y_pred_inverse[i]}")

    # Visualizar as previsões
    for i in range(len(important_points)):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inverse[:, i], label='Real')
        plt.plot(y_pred_inverse[:, i], label='Predito')
        plt.title(f'Comparação: {important_points[i]}')
        plt.xlabel('Amostras')
        plt.ylabel('Valor')
        plt.legend()
        plt.savefig(f'deep_learning_{important_points[i]}.png')
        plt.close()

# Exemplo de uso:
# df_imu = pd.read_csv('dados_imu.csv')
# df_gold = pd.read_csv('dados_padro_ouro.csv')
# deep_learning(df_imu, df_gold)
