import pandas as pd
import numpy as np

class FeatureExtraction:
    """
    A class for extracting and analyzing features from wearable and camera data.
    """
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

    @staticmethod
    def calculate_velocity(acc_series, dt):
        """
        Calculate velocity by integrating acceleration data.

        Parameters
        ----------
        acc_series : pd.Series
            Series of acceleration data.
        dt : float
            Time interval between data points.

        Returns
        -------
        pd.Series
            Series of calculated velocity data.
        """
        return acc_series.cumsum() * dt

    @staticmethod
    def calculate_acceleration(vel_series, dt):
        """
        Calculate acceleration by differentiating velocity data.

        Parameters
        ----------
        vel_series : pd.Series
            Series of velocity data.
        dt : float
            Time interval between data points.

        Returns
        -------
        pd.Series
            Series of calculated acceleration data.
        """
        return vel_series.diff() / dt

    @staticmethod
    def calculate_angular_acceleration(angular_velocity, dt):
        """
        Calcula a aceleração angular a partir da velocidade angular usando diferenciação numérica.
        
        Parameters
        ----------
        angular_velocity : pd.Series
            Série temporal de velocidade angular (giroscópio).
        dt : float
            Intervalo de tempo entre as medições.

        Returns
        -------
        pd.Series
            Série temporal da aceleração angular calculada.
        """
        angular_acceleration = np.diff(angular_velocity) / dt
        mean_value = np.mean(angular_acceleration)
        angular_acceleration = np.insert(angular_acceleration, 0, mean_value)
        return angular_acceleration

    @staticmethod
    def calculate_magnitude(x, y, z):
        """
        Calcula a magnitude de um vetor 3D a partir de suas componentes x, y e z.

        Parameters
        ----------
        x : pd.Series
            Componente x do vetor.
        y : pd.Series
            Componente y do vetor.
        z : pd.Series
            Componente z do vetor.

        Returns
        -------
        pd.Series
            Série temporal da magnitude do vetor.
        """
        return np.sqrt(x**2 + y**2 + z**2)

    @staticmethod
    def calculate_jerk(acc_series, dt):
        """
        Calcula a magnitude de um vetor 3D a partir de suas componentes x, y e z.

        Parameters
        ----------
        x : pd.Series
            Componente x do vetor.
        y : pd.Series
            Componente y do vetor.
        z : pd.Series
            Componente z do vetor.

        Returns
        -------
        pd.Series
            Série temporal da magnitude do vetor.
        """
        jerk = np.diff(acc_series) / dt
        return np.insert(jerk, 0, 0)

    @staticmethod
    def calculate_linear_position(vel_series, dt):
        """
        Calcula a magnitude de um vetor 3D a partir de suas componentes x, y e z.

        Parameters
        ----------
        x : pd.Series
            Componente x do vetor.
        y : pd.Series
            Componente y do vetor.
        z : pd.Series
            Componente z do vetor.

        Returns
        -------
        pd.Series
            Série temporal da magnitude do vetor.
        """
        pos = np.cumsum(vel_series[:-1] * dt)
        return np.insert(pos, 0, 0)