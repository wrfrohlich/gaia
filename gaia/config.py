from os import path

class Config():
    def __init__(self, participant="040102"):
        self.path_project_gaia = path.dirname(__file__)
        self.path_project_gaia_parent = path.dirname(self.path_project_gaia)
        self.path_parent = path.dirname(self.path_project_gaia_parent)
        self.data_path = path.join(self.path_parent, "bittencourt_data")

        self.gait_lab_point_file = path.join(self.data_path, f"point_track_{participant}.emt")
        self.gait_lab_force_file = path.join(self.data_path, f"force_track_{participant}.emt")
        self.gwalk_file =  path.join(self.data_path, f"gwalk_{participant}.txt")

        self.experiments = path.join(self.path_project_gaia_parent, "experiments")

        self.scalars = {
            'roll', 'pitch', 'yaw'
        }

        self.vectors = {
            "acc", "gyro", "c7",
            "r_should", "l_should",
            "sacrum_s", "r_asis", "l_asis", "MIDASIS",
            "r_knee_1", "l_knee_1",
            "r_mall", "l_mall",
            "r_heel", "l_heel", "r_met", "l_met",
            "PO", "SHO"
        }

        self.body_parts = {
            "imu": [
                'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw',
                'vel_x', 'vel_y', 'vel_z',
                'ang_acc_gyro_x', 'ang_acc_gyro_y', 'ang_acc_gyro_z',
                'mag_acc', 'mag_gyro',
                'jerk_x', 'jerk_x', 'jerk_x',
                'pos_x', 'pos_y', 'pos_z'],

            "head_neck": ["c7_x", "c7_y", "c7_z"],
            
            "upper_body": [
                "r_should_x", "r_should_y", "r_should_z",
                "l_should_x", "l_should_y", "l_should_z"
            ],

            "trunk": [
                "sacrum_s_x", "sacrum_s_y", "sacrum_s_z",
                "r_asis_x", "r_asis_y", "r_asis_z",
                "l_asis_x", "l_asis_y", "l_asis_z",
                "MIDASIS_x", "MIDASIS_y", "MIDASIS_z"
            ],

            "upper_legs": [
                "r_knee_1_x", "r_knee_1_y", "r_knee_1_z",
                "l_knee_1_x", "l_knee_1_y", "l_knee_1_z"
            ],

            "lower_legs": [
                "r_mall_x", "r_mall_y", "r_mall_z",
                "l_mall_x", "l_mall_y", "l_mall_z"
            ],

            "feet": [
                "r_heel_x", "r_heel_y", "r_heel_z",
                "l_heel_x", "l_heel_y", "l_heel_z",
                "r_met_x", "r_met_y", "r_met_z",
                "l_met_x", "l_met_y", "l_met_z"
            ],

            "additional_points": [
                "SHO_x", "SHO_y", "SHO_z",
                "PO_x", "PO_y", "PO_z"
            ],

            "force": [
                'r_force_x', 'r_force_y', 'r_force_z',
                'l_force_x', 'l_force_y', 'l_force_z'
            ]
        }