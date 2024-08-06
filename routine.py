from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.preprocessing import Preprocessing
from gaia.feature_extraction import FeatureExtraction
from gaia.machine_learning import prediction
from gaia.deep_learning import deep_learning


class Routine():
    def __init__(self):
        self.config = Config()
        self.path = self.config.data_path
        self.gait_lab_file = self.config.gait_lab_file
        self.gwalk_file = self.config.gwalk_file

        self.gwalkrecords = ["gwalk"]
        self.gait_lab_records = ["force_track", "point_track", "torque_track"]
        self.participants = ["040102", "040103", "040104", "040105"]

    def run_experiment01(self):
        """
        Correlation - Raw data removing NaN data
        """
        name = "experiment01"

        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Preprocessing()
        merged_data = proc.merge(gait_lab_data, gwalk_data)

    def run_experiment02(self):
        name = "experiment02"

    def run_experiment03(self):
        name = "experiment03"

    def run_correlation(self):


        corr = Correlation()
        corr.corr_matrix(merged_data)

    def run_feature_extraction(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Preprocessing()
        merged_data = proc.run(gait_lab_data, gwalk_data)

        feat_ext = FeatureExtraction()
        feat_ext.feature_extraction(merged_data, gwalk_data)

    def run_machine_learning(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Preprocessing()
        gait_lab_data = proc.preprocessing(gait_lab_data)
        gwalk_data = proc.preprocessing(gwalk_data)

        prediction(gwalk_data, gait_lab_data)

    def run_deep_learning(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Preprocessing()
        gait_lab_data = proc.preprocessing(gait_lab_data)
        gwalk_data = proc.preprocessing(gwalk_data)

        print(gait_lab_data)

        deep_learning(gwalk_data, gait_lab_data)


if __name__ == '__main__':
    routine = Routine()
    routine.run_deep_learning()
    #routine.run_correlation()
    #routine.run_feature_extraction()
    #routine.run_machine_learning()