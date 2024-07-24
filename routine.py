from load import GaitLab, GWalk
from correlation import Correlation
from processing import Processing

class Routine():
    def __init__(self):
        self.path = "../bittencourt_data/"
        self.gwalkrecords = ["gwalk"]
        self.gait_lab_records = ["force_track", "point_track", "torque_track"]
        self.participants = ["040102", "040103", "040104", "040105"]

    def run(self):
        #gait_lab_file = "../bittencourt_data/point_track_040102.emt"
        gait_lab_file = "../bittencourt_data/force_track_040102.emt"
        gwalk_file = "../bittencourt_data/gwalk_040102.txt"

        gl = GaitLab()
        gait_lab_data = gl.load_data(gait_lab_file)
        print(gait_lab_data)

        gw = GWalk()
        gwalk_data = gw.load_data(gwalk_file)
        print(gwalk_data)

        proc = Processing()
        merged_data = proc.run(gait_lab_data, gwalk_data)

        Corr = Correlation()
        Corr.corr_matrix(merged_data)


if __name__ == '__main__':
    routine = Routine()
    routine.run()