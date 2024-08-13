from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.clustering import Clustering
from gaia.preprocessing import Preprocessing
from gaia.feature_extraction import FeatureExtraction


class Routine():
    def __init__(self):
        self.config = Config()
        self.path = self.config.data_path
        self.gwalk_file = self.config.gwalk_file
        self.gait_lab_force_file = self.config.gait_lab_force_file
        self.gait_lab_point_file = self.config.gait_lab_point_file

        self.gwalkrecords = ["gwalk"]
        self.gait_lab_records = ["force_track", "point_track", "torque_track"]
        self.participants = ["040102", "040103", "040104", "040105"]

        gl = GaitLab()
        self.df1 = gl.load_data(self.gait_lab_force_file)
        self.df2 = gl.load_data(self.gait_lab_point_file)

        gw = GWalk()
        self.df3 = gw.load_data(self.gwalk_file)

    def run_experiment01(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
        """
        name = "experiment21"
        preproc = Preprocessing()
        dt = self.df3['time'].diff().mean()
        self.df3['vel_x'] = FeatureExtraction.calculate_velocity(self.df3['acc_x'], dt)
        self.df3['vel_y'] = FeatureExtraction.calculate_velocity(self.df3['acc_y'], dt)
        self.df3['vel_z'] = FeatureExtraction.calculate_velocity(self.df3['acc_z'], dt)

        self.df3['ang_acc_gyro_x'] = FeatureExtraction.calculate_angular_acceleration(self.df3['gyro_x'], dt)
        self.df3['ang_acc_gyro_y'] = FeatureExtraction.calculate_angular_acceleration(self.df3['gyro_y'], dt)
        self.df3['ang_acc_gyro_z'] = FeatureExtraction.calculate_angular_acceleration(self.df3['gyro_z'], dt)

        self.df3['mag_acc'] = FeatureExtraction.calculate_magnitude(self.df3['acc_x'], self.df3['acc_y'], self.df3['acc_z'])
        self.df3['mag_gyro'] = FeatureExtraction.calculate_magnitude(self.df3['gyro_x'], self.df3['gyro_y'], self.df3['gyro_z'])
        merged_data = preproc.run(self.df1, self.df2, self.df3, remove_nan=True, convert_nan="mean", interpolate_method="linear", filter_data="low-pass", normalization="minmax")

        clust = Clustering(name=name)
        clust.run_kmeans(merged_data)
        #corr = Correlation(name=name)
        #corr.intraclass_correlation_exploratory(merged_data)

if __name__ == '__main__':
    routine = Routine()
    routine.run_experiment01()