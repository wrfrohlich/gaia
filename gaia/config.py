from os import path

class Config():
    def __init__(self):
        self.path_project_gaia = path.dirname(__file__)
        self.path_project_gaia_parent = path.dirname(self.path_project_gaia)
        self.path_parent = path.dirname(self.path_project_gaia_parent)
        self.data_path = path.join(self.path_parent, "bittencourt_data")

        self.gait_lab_file = path.join(self.data_path, "point_track_040102.emt")
        #self.gait_lab_file = path.join(self.data_path, "force_track_040102.emt")
        self.gwalk_file =  path.join(self.data_path, "gwalk_040102.txt")

        self.figures = path.join(self.path_project_gaia_parent, "figures")

        self.body_parts = {
            "imu": [
                'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw'],

            "head_neck": ["time", "c7_x", "c7_y", "c7_z"],
            
            "upper_body": [
                "r should_x", "r should_y", "r should_z",
                "l should_x", "l should_y", "l should_z"
            ],

            "trunk": [
                "sacrum s_x", "sacrum s_y", "sacrum s_z",
                "r asis_x", "r asis_y", "r asis_z",
                "l asis_x", "l asis_y", "l asis_z",
                "MIDASIS_x", "MIDASIS_y", "MIDASIS_z"
            ],

            "upper_legs": [
                "r knee 1_x", "r knee 1_y", "r knee 1_z",
                "l knee 1_x", "l knee 1_y", "l knee 1_z"
            ],

            "lower_legs": [
                "r mall_x", "r mall_y", "r mall_z",
                "l mall_x", "l mall_y", "l mall_z"
            ],

            "feet": [
                "r heel_x", "r heel_y", "r heel_z",
                "l heel_x", "l heel_y", "l heel_z",
                "r met_x", "r met_y", "r met_z",
                "l met_x", "l met_y", "l met_z"
            ],

            "additional_points": [
                "SHO_x", "SHO_y", "SHO_z",
                "PO_x", "PO_y", "PO_z"
            ]
        }