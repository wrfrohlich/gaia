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
        self.path_corr_matrix = path.join(self.figures, "mean")
        self.path_corr_cross = path.join(self.figures, "cross")