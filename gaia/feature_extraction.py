import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.models import Model
from keras.layers import Dense, Conv1D, Flatten, Reshape, Input, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway


class FeatureExtraction:
    """
    A class for extracting and analyzing features from wearable and camera data.
    """

    def feature_extraction(self, merged_data, wearable_data):
        """
        Extracts features from wearable and camera data, applies K-Means clustering, and performs analysis and visualization.

        Parameters
        ----------
        merged_data : pd.DataFrame
            A DataFrame containing the merged wearable and camera data, with columns for wearable sensor readings 
            and camera points.
        wearable_data : pd.DataFrame
            A DataFrame containing the wearable data with timestamps and other relevant columns for ANOVA analysis.

        Returns
        -------
        None
            This method does not return any values. It performs feature extraction, clustering, and generates plots.
        """
        
        # Normalize wearable and camera data
        scaler = StandardScaler()
        wearable_scaled = scaler.fit_transform(merged_data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])
        camera_scaled = scaler.fit_transform(merged_data.drop(columns=['time']))

        # Prepare data for the autoencoder
        X_wearable = wearable_scaled.reshape((wearable_scaled.shape[0], wearable_scaled.shape[1], 1))

        # Build and compile the autoencoder model
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

        # Train the autoencoder with wearable data
        autoencoder.fit(X_wearable, X_wearable, epochs=20, verbose=1)

        # Extract features using the encoder
        wearable_features = encoder.predict(X_wearable)

        # Apply PCA to reduce dimensionality of camera data
        pca = PCA(n_components=10)
        camera_features = pca.fit_transform(camera_scaled)

        # Combine extracted features
        combined_features = np.concatenate((wearable_features, camera_features), axis=1)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(combined_features)

        # Add cluster labels to the original data
        merged_data['cluster'] = clusters

        # Visualize clusters in the original feature space
        sns.scatterplot(x='acc_x', y='r should.X', hue='cluster', data=merged_data, palette='viridis')
        plt.title('Clusters between Wearable and Kinematic Data')
        plt.savefig('Clusters.png')
        plt.clf()

        # Calculate and print the silhouette coefficient
        silhouette_avg = silhouette_score(combined_features, clusters)
        print(f'Silhouette Coefficient: {silhouette_avg}')

        # Perform ANOVA analysis
        anova_results = {}
        for col in wearable_data.columns:
            if col != 'timestamp':
                groups = [merged_data[merged_data['cluster'] == i][col] for i in range(3)]
                anova_results[col] = f_oneway(*groups)
                print(f'{col} - ANOVA: F-value={anova_results[col].statistic}, p-value={anova_results[col].pvalue}')

        # Visualization with PCA (2D projection for clustering)
        pca_2d = PCA(n_components=2)
        combined_2d = pca_2d.fit_transform(combined_features)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=combined_2d[:, 0], y=combined_2d[:, 1], hue=clusters, palette='viridis')
        plt.title('Clusters in 2D using PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig('Clusters_PCA.png')
        plt.clf()

    @staticmethod
    def distance_between_points(df):
        """
        Calculate the Euclidean distance between the right and left heel positions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns for right and left heel positions.

        Returns
        -------
        None
            Adds a new column 'dist_heel' to the DataFrame.
        """
        df['dist_heel'] = np.sqrt((df['r_heel_x'] - df['l_heel_x'])**2 + 
                                  (df['r_heel_y'] - df['l_heel_y'])**2 + 
                                  (df['r_heel_z'] - df['l_heel_z'])**2)
    
    @staticmethod
    def calculate_movement_speed(df):
        """
        Calculate the speed of movement for the right heel.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns for right heel positions and time.

        Returns
        -------
        None
            Adds a new column 'r_heel_speed' to the DataFrame.
        """
        df['r_heel_speed'] = np.sqrt(df['r_heel_x'].diff()**2 + 
                                     df['r_heel_y'].diff()**2 + 
                                     df['r_heel_z'].diff()**2) / df['time'].diff()
        
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate the angle between three points.

        Parameters
        ----------
        p1 : np.array
            Coordinates of the first point.
        p2 : np.array
            Coordinates of the second point (vertex of the angle).
        p3 : np.array
            Coordinates of the third point.

        Returns
        -------
        float
            The angle in degrees.
        """
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def calculate_angle_between_segments(df):
        """
        Calculate the angle between segments formed by right shoulder, right knee, and right heel positions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns for right shoulder, knee, and heel positions.

        Returns
        -------
        None
            Adds a new column 'knee_angle' to the DataFrame.
        """
        angles = []
        for i in range(len(df)):
            p1 = np.array([df.loc[i, 'r_shoulder_x'], df.loc[i, 'r_shoulder_y'], df.loc[i, 'r_shoulder_z']])
            p2 = np.array([df.loc[i, 'r_knee_x'], df.loc[i, 'r_knee_y'], df.loc[i, 'r_knee_z']])
            p3 = np.array([df.loc[i, 'r_heel_x'], df.loc[i, 'r_heel_y'], df.loc[i, 'r_heel_z']])
            angles.append(FeatureExtraction.calculate_angle(p1, p2, p3))
        df['knee_angle'] = angles
