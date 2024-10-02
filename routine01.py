from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.preprocessing import Preprocessing


class Routine:
    def __init__(self):
        self.config = Config()
        self.path = self.config.data_path
        self.gait_lab_file = self.config.gait_lab_point_file
        self.gwalk_file = self.config.gwalk_file

        self.gwalkrecords = ["gwalk"]
        self.gait_lab_records = ["force_track", "point_track", "torque_track"]
        self.participants = ["040102", "040103", "040104", "040105"]

        gl = GaitLab()
        self.df1 = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        self.df2 = gw.load_data(self.gwalk_file)

    def run_experiment01(self):
        """
        Correlation:
            - Raw data removing NaN data
        """
        name = "experiment01"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan=None,
            interpolate_method=None,
            filter_data=None,
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment02(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into zero
        """
        name = "experiment02"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="zero",
            interpolate_method=None,
            filter_data=None,
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment03(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
        """
        name = "experiment03"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method=None,
            filter_data=None,
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment04(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Interpolation
        """
        name = "experiment04"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan=None,
            interpolate_method="linear",
            filter_data=None,
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment05(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
        """
        name = "experiment05"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data=None,
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment06(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
        """
        name = "experiment06"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="low-pass",
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment07(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Band-Pass
        """
        name = "experiment07"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="band-pass",
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment08(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth High-Pass
        """
        name = "experiment08"
        preproc = Preprocessing()
        merged_data = preproc.run(
            self.df1,
            self.df2,
            remove_nan=True,
            convert_nan="mean",
            interpolate_method="linear",
            filter_data="high-pass",
            normalization=None,
        )
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment09(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
        """
        name = "experiment09"
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
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment10(self):
        """
        Correlation:
            - Raw data removing NaN data
            - Convert NaN into the mean of all the values
            - Interpolation
            - Filtering Butterworth Low-Pass
            - Normalization
            - Vectors Magnitude
        """
        name = "experiment10"
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
        data = preproc.get_magnitude(merged_data)
        corr = Correlation(name=name)
        corr.corr_matrix_special(data, name="magnitude")


if __name__ == "__main__":
    routine = Routine()
    routine.run_experiment01()
    routine.run_experiment02()
    routine.run_experiment03()
    routine.run_experiment04()
    routine.run_experiment05()
    routine.run_experiment06()
    routine.run_experiment07()
    routine.run_experiment08()
    routine.run_experiment09()
    routine.run_experiment10()
