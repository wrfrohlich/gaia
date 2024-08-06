from os import path

class Config():
    def __init__(self):
        self.path_project_gaia = path.dirname(__file__)
        self.path_parent = path.dirname(path.dirname(self.path_project_gaia))
        self.data_path = path.join(self.path_parent, "bittencourt_data")

        self.gait_lab_file = path.join(self.data_path, "point_track_040102.emt")
        #self.gait_lab_file = path.join(self.data_path, "force_track_040102.emt")
        self.gwalk_file =  path.join(self.data_path, "gwalk_040102.txt")

        self.figures = path.join(self.path_project_gaia, "figures")

        data_upper = [[
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r should_x", "r should_y", "r should_z",
            "l should_x", "l should_y", "l should_z",
            "sacrum s_x", "sacrum s_y", "sacrum s_z",
            "PO_x", "PO_y", "PO_z"
        ]]
        data_lower_01 = [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r knee 1_x", "r knee 1_y", "r knee 1_z",
            "l knee 1_x", "l knee 1_y", "l knee 1_z",
            "r mall_x", "r mall_y", "r mall_z",
            "l mall_x", "l mall_y", "l mall_z"
        ]
        data_lower_02 = [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "roll", "pitch", "yaw",
            "r heel_x", "r heel_y", "r heel_z",
            "l heel_x", "l heel_y", "l heel_z",
            "r met_x", "r met_y", "r met_z",
            "l met_x", "l met_y", "l met_z"
        ]