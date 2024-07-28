from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.processing import Processing
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

    def run_correlation(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Processing()
        merged_data = proc.run(gait_lab_data, gwalk_data)

        corr = Correlation()
        corr.corr_matrix(merged_data)

    def run_feature_extraction(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Processing()
        merged_data = proc.run(gait_lab_data, gwalk_data)

        feat_ext = FeatureExtraction()
        feat_ext.feature_extraction(merged_data, gwalk_data)


if __name__ == '__main__':
    routine = Routine()
    routine.run_correlation()
    routine.run_feature_extraction()