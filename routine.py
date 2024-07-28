from gaia.config import Config
from gaia.load import GaitLab, GWalk
from gaia.correlation import Correlation
from gaia.processing import Processing

class Routine():
    def __init__(self):
        self.config = Config()
        self.path = self.config.data_path
        self.gait_lab_file = self.config.gait_lab_file
        self.gwalk_file = self.config.gwalk_file

        self.gwalkrecords = ["gwalk"]
        self.gait_lab_records = ["force_track", "point_track", "torque_track"]
        self.participants = ["040102", "040103", "040104", "040105"]

    def run(self):
        gl = GaitLab()
        gait_lab_data = gl.load_data(self.gait_lab_file)

        gw = GWalk()
        gwalk_data = gw.load_data(self.gwalk_file)

        proc = Processing()
        merged_data = proc.run(gait_lab_data, gwalk_data)

        #gwalk_data = proc.remove_nan(gwalk_data)
        #gait_lab_data = proc.remove_nan(gait_lab_data)

        corr = Correlation()
        #corr.cross_correlation(merged_data)
        corr.corr_matrix(merged_data)


if __name__ == '__main__':
    routine = Routine()
    routine.run()