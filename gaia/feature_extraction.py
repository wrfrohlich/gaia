import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, Input, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway


class FeatureExtraction():
    def feature_extraction(self, merged_data, wearable_data):

        # Normalizar os dados
        scaler = StandardScaler()
        wearable_scaled = scaler.fit_transform(merged_data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])
        camera_scaled = scaler.fit_transform(merged_data.drop(columns=['time']))

        # Preparar dados para o autoencoder
        X_wearable = wearable_scaled.reshape((wearable_scaled.shape[0], wearable_scaled.shape[1], 1))

        # Construir o autoencoder
        input_layer = Input(shape=(X_wearable.shape[1], 1))
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        encoded = Dense(50, activation='relu')(x)
        decoded = Dense(X_wearable.shape[1], activation='relu')(encoded)
        decoded = Reshape((X_wearable.shape[1], 1))(decoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        autoencoder.compile(optimizer='adam', loss='mse')

        # Treinar o autoencoder com os dados dos wearables
        autoencoder.fit(X_wearable, X_wearable, epochs=20, verbose=1)

        # Extrair características dos dados dos wearables usando o encoder
        wearable_features = encoder.predict(X_wearable)

        # Aplicar PCA para reduzir a dimensionalidade dos dados cinemáticos
        pca = PCA(n_components=10)
        camera_features = pca.fit_transform(camera_scaled)

        # Combinar características extraídas
        combined_features = np.concatenate((wearable_features, camera_features), axis=1)

        # Aplicar K-Means para encontrar padrões comuns
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(combined_features)

        # Adicionar clusters aos dados originais
        merged_data['cluster'] = clusters

        # Visualizar clusters
        sns.scatterplot(x='acc_x', y='r should.X', hue='cluster', data=merged_data, palette='viridis')
        plt.title('Clusters entre Dados de Wearables e Cinemática')
        plt.show()

        # Calcular o coeficiente de silhueta
        silhouette_avg = silhouette_score(combined_features, clusters)
        print(f'Coeficiente de Silhueta: {silhouette_avg}')

        # Análise de variância (ANOVA)
        anova_results = {}
        for col in wearable_data.columns:
            if col != 'timestamp':
                groups = [merged_data[merged_data['cluster'] == i][col] for i in range(3)]
                anova_results[col] = f_oneway(*groups)
                print(f'{col} - ANOVA: F-value={anova_results[col].statistic}, p-value={anova_results[col].pvalue}')

        # Visualização com PCA (reduzindo para 2D para visualização)
        pca_2d = PCA(n_components=2)
        combined_2d = pca_2d.fit_transform(combined_features)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=combined_2d[:, 0], y=combined_2d[:, 1], hue=clusters, palette='viridis')
        plt.title('Clusters em 2D usando PCA')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.savefig(f'bluba_2.png')