from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.preprocessing import Preprocessing
from gaia.feature_extraction import FeatureExtraction


class Routine:
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
        name = "experiment11"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        merged_data = preproc.run(
            merged_data,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment02(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment12"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        merged_data = preproc.run(
            merged_data,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        corr = Correlation(name=name)
        res_corr = corr.get_higher_corr(merged_data, level="0.4")
        corr.analyze_correlation()

    def run_experiment03(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment13"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        merged_data = preproc.run(
            merged_data,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level="0.4")
        corr_data = corr.analyze_correlation()
        corr.cross_correlation(merged_data, corr_data)

    def run_experiment04(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment14"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        merged_data = preproc.run(
            merged_data,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level="0.4")
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data)

    def run_experiment05(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment15"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        merged_data = preproc.run(
            merged_data,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="standard",
        )
        corr = Correlation(name=name)
        corr.cross_correlation_exploratory(merged_data)

    def run_experiment06(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment16"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="minmax",
        )
        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level="0.4")
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data)
        corr.cross_correlation_exploratory(merged_data, criterion=0.4, best=True)

    def run_experiment07(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment17"
        preproc = Preprocessing()
        dt = self.df3["time"].diff().mean()
        self.df3["vel_x"] = FeatureExtraction.calculate_velocity(self.df3["acc_x"], dt)
        self.df3["vel_y"] = FeatureExtraction.calculate_velocity(self.df3["acc_y"], dt)
        self.df3["vel_z"] = FeatureExtraction.calculate_velocity(self.df3["acc_z"], dt)

        self.df3["ang_acc_gyro_x"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_x"], dt
        )
        self.df3["ang_acc_gyro_y"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_y"], dt
        )
        self.df3["ang_acc_gyro_z"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_z"], dt
        )

        self.df3["mag_acc"] = FeatureExtraction.calculate_magnitude(
            self.df3["acc_x"], self.df3["acc_y"], self.df3["acc_z"]
        )
        self.df3["mag_gyro"] = FeatureExtraction.calculate_magnitude(
            self.df3["gyro_x"], self.df3["gyro_y"], self.df3["gyro_z"]
        )
        merged_data = preproc.run(
            self.df1,
            self.df2,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="minmax",
        )

        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level=0.4)
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data)
        corr.cross_correlation_exploratory(merged_data, criterion=0.4, best=True)

    def run_experiment08(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment18"
        preproc = Preprocessing()
        dt = self.df3["time"].diff().mean()
        self.df3["vel_x"] = FeatureExtraction.calculate_velocity(self.df3["acc_x"], dt)
        self.df3["vel_y"] = FeatureExtraction.calculate_velocity(self.df3["acc_y"], dt)
        self.df3["vel_z"] = FeatureExtraction.calculate_velocity(self.df3["acc_z"], dt)

        self.df3["ang_acc_gyro_x"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_x"], dt
        )
        self.df3["ang_acc_gyro_y"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_y"], dt
        )
        self.df3["ang_acc_gyro_z"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_z"], dt
        )

        self.df3["mag_acc"] = FeatureExtraction.calculate_magnitude(
            self.df3["acc_x"], self.df3["acc_y"], self.df3["acc_z"]
        )
        self.df3["mag_gyro"] = FeatureExtraction.calculate_magnitude(
            self.df3["gyro_x"], self.df3["gyro_y"], self.df3["gyro_z"]
        )
        merged_data = preproc.run(
            self.df1,
            self.df2,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="minmax",
        )

        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level=0.7)
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data)
        corr.cross_correlation_exploratory(merged_data, criterion=0.7, best=True)

    def run_experiment09(self):
        """
        Cross-correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment19"
        preproc = Preprocessing()
        dt = self.df3["time"].diff().mean()
        self.df3["vel_x"] = FeatureExtraction.calculate_velocity(self.df3["acc_x"], dt)
        self.df3["vel_y"] = FeatureExtraction.calculate_velocity(self.df3["acc_y"], dt)
        self.df3["vel_z"] = FeatureExtraction.calculate_velocity(self.df3["acc_z"], dt)

        self.df3["ang_acc_gyro_x"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_x"], dt
        )
        self.df3["ang_acc_gyro_y"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_y"], dt
        )
        self.df3["ang_acc_gyro_z"] = FeatureExtraction.calculate_angular_acceleration(
            self.df3["gyro_z"], dt
        )

        self.df3["mag_acc"] = FeatureExtraction.calculate_magnitude(
            self.df3["acc_x"], self.df3["acc_y"], self.df3["acc_z"]
        )
        self.df3["mag_gyro"] = FeatureExtraction.calculate_magnitude(
            self.df3["gyro_x"], self.df3["gyro_y"], self.df3["gyro_z"]
        )
        merged_data = preproc.run(
            self.df1,
            self.df2,
            self.df3,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization="minmax",
        )

        print(merged_data)
        corr = Correlation(name=name)
        corr.get_higher_corr(merged_data, level=0.5)
        corr_data = corr.analyze_correlation()
        corr.cross_correlation_analysis(merged_data, corr_data)
        corr.cross_correlation_exploratory(merged_data, criterion=0.5, best=True)


if __name__ == "__main__":
    routine = Routine()
    routine.run_experiment01()
    routine.run_experiment02()
    # routine.run_experiment03()
    # routine.run_experiment04()
    # routine.run_experiment05()
    # routine.run_experiment06()
    # routine.run_experiment07()
    # routine.run_experiment08()
    # routine.run_experiment09()
