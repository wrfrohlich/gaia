import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def deep_learning(df_imu, df_gold):

    # Sincronizar os dados pelo tempo
    df_combined = pd.merge_asof(df_imu, df_gold, on='time', direction='nearest')

    # Selecionar os pontos cinéticos mais importantes
    important_points = [
        'sacrum_x', 'sacrum_y', 'sacrum_z',
        'r asis_x', 'r asis_y', 'r asis_z', 'l asis_x', 'l asis_y', 'l asis_z',
        'r knee 1_x', 'r knee 1_y', 'r knee 1_z', 'l knee 1_x', 'l knee 1_y', 'l knee 1_z',
        'r mall_x', 'r mall_y', 'r mall_z', 'l mall_x', 'l mall_y', 'l mall_z',
        'r heel_x', 'r heel_y', 'r heel_z', 'l heel_x', 'l heel_y', 'l heel_z',
        'r met_x', 'r met_y', 'r met_z', 'l met_x', 'l met_y', 'l met_z'
    ]

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

    # Construir o modelo LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Avaliar o modelo
    loss = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')

    # Previsões
    y_pred = model.predict(X_test)

    # Inverter a normalização das previsões para comparação
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Comparar previsões com o padrão ouro
    for i in range(len(y_test_inverse)):
        print(f"Real: {y_test_inverse[i]}, Predito: {y_pred_inverse[i]}")

    import matplotlib.pyplot as plt

    # Previsões
    y_pred = model.predict(X_test)

    # Inverter a normalização das previsões para comparação
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    # Comparar previsões com o padrão ouro
    for i in range(len(important_points)):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_inverse[:, i], label='Real')
        plt.plot(y_pred_inverse[:, i], label='Predito')
        plt.title(f'Comparação: {important_points[i]}')
        plt.xlabel('Amostras')
        plt.ylabel('Valor')
        plt.legend()
        plt.savefig('deep_learning.png')
        plt.clf()