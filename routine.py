from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.preprocessing import Preprocessing
from gaia.feature_extraction import FeatureExtraction


class Routine():
    def __init__(self):
        self.config = Config()
        self.path = self.config.data_path
        self.gait_lab_file = self.config.gait_lab_file
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
        Correlation - Raw data removing NaN data
        """
        name = "experiment01"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan=None, interpolate_method=None, filter_data=None, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment02(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment02"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan="zero", interpolate_method=None, filter_data=None, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment03(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment03"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan="mean", interpolate_method=None, filter_data=None, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment04(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment04"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan=None, interpolate_method="linear", filter_data=None, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment05(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment05"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan="mean", interpolate_method=None, filter_data=True, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment06(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment06"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan=None, interpolate_method="linear", filter_data=True, normalization=None)
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)

    def run_experiment07(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment07"
        preproc = Preprocessing()
        merged_data = preproc.run(self.df1, self.df2, remove_nan=True, convert_nan=None, interpolate_method="linear", filter_data=True, normalization="standard")
        corr = Correlation(name=name)
        corr.corr_matrix(merged_data)



if __name__ == '__main__':
    routine = Routine()
    routine.run_experiment01()
    routine.run_experiment02()
    routine.run_experiment03()
    routine.run_experiment04()
    routine.run_experiment05()
    routine.run_experiment06()
    routine.run_experiment07()