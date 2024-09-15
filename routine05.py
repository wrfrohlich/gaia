from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.clustering import Clustering
from gaia.preprocessing import Preprocessing
from gaia.feature_extraction import FeatureExtraction
from gaia.machine_learning import MachineLearningvi 


class Routine():
    def __init__(self):
        self.config = Config(participant="040102")
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
        name = "experiment31"
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

        self.df3['jerk_x'] = FeatureExtraction.calculate_jerk(self.df3['acc_x'], dt)
        self.df3['jerk_y'] = FeatureExtraction.calculate_jerk(self.df3['acc_y'], dt)
        self.df3['jerk_z'] = FeatureExtraction.calculate_jerk(self.df3['acc_z'], dt)

        self.df3['pos_x'] = FeatureExtraction.calculate_jerk(self.df3['vel_x'], dt)
        self.df3['pos_y'] = FeatureExtraction.calculate_jerk(self.df3['vel_y'], dt)
        self.df3['pos_z'] = FeatureExtraction.calculate_jerk(self.df3['vel_z'], dt)

        merged_data = preproc.run(self.df1, self.df2, self.df3, remove_nan=True, convert_nan="mean", interpolate_method="linear", filter_data="low-pass", normalization="minmax")

        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level=0.4)
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data, print_fig=False)
        corr.cross_correlation_exploratory(merged_data, criterion=0.4, best=True, print_fig=False)

        clust = Clustering(name=name)
        clust.analyze_cross_correlation()
        clust.run_clustering_kmeans_grouped(merged_data, method="pca_shift", print_fig=False)

        ml = MachineLearning(name=name)
        sep_data = ml.sep_data()
        ml.prep_data(sep_data, merged_data)


if __name__ == '__main__':
    routine = Routine()
    routine.run_experiment01()