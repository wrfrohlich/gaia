from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.preprocessing import Preprocessing


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
            - Vectors Magnitude
        """
        name = "experiment11"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan="mean", interpolate_method="linear", filter_data="low-pass", normalization="standard")
        merged_data = preproc.run(merged_data, self.df3, remove_nan=True, convert_nan="mean", interpolate_method="linear", filter_data="low-pass", normalization="standard")
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

if __name__ == '__main__':
    routine = Routine()
    routine.run_experiment01()