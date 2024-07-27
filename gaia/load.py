"""
Load Gait files
"""

import math
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GaitLab():
    def __init__(self):
        self.data_force_track = [
            'frame', 'time', 'rx', 'ry', 'rz', 'lx', 'ly', 'lz'
        ]
        self.data_point_track = [
            "frame", "time", "c7.X", "c7.Y", "c7.Z", "r should.X", "r should.Y",
            "r should.Z", "l should.X", "l should.Y", "l should.Z", "sacrum.X",
            "sacrum.Y", "sacrum.Z", "r asis.X", "r asis.Y", "r asis.Z", "r bar 1.X",
            "r bar 1.Y", "r bar 1.Z", "r knee 1.X", "r knee 1.Y", "r knee 1.Z",
            "r bar 2.X", "r bar 2.Y", "r bar 2.Z", "r mall.X", "r mall.Y", "r mall.Z",
            "r heel.X", "r heel.Y", "r heel.Z", "r met.X", "r met.Y", "r met.Z",
            "l asis.X", "l asis.Y", "l asis.Z", "l bar 1.X", "l bar 1.Y", "l bar 1.Z",
            "l knee 1.X", "l knee 1.Y", "l knee 1.Z", "l bar 2.X", "l bar 2.Y",
            "l bar 2.Z", "l mall.X", "l mall.Y", "l mall.Z", "l heel.X", "l heel.Y",
            "l heel.Z", "l met.X", "l met.Y", "l met.Z", "l asis s.X", "l asis s.Y",
            "l asis s.Z", "l bar 1 s.X", "l bar 1 s.Y", "l bar 1 s.Z", "l knee 1 s.X",
            "l knee 1 s.Y", "l knee 1 s.Z", "l bar 2 s.X", "l bar 2 s.Y", "l bar 2 s.Z",
            "l mall s.X", "l mall s.Y", "l mall s.Z", "sacrum s.X", "sacrum s.Y",
            "sacrum s.Z", "r asis s.X", "r asis s.Y", "r asis s.Z", "r bar 1 s.X",
            "r bar 1 s.Y", "r bar 1 s.Z", "r knee 1 s.X", "r knee 1 s.Y",
            "r knee 1 s.Z", "r bar 2 s.X", "r bar 2 s.Y", "r bar 2 s.Z", "r mall s.X",
            "r mall s.Y", "r mall s.Z", "MIDASIS.X", "MIDASIS.Y", "MIDASIS.Z", 
            "r asis s2.X", "r asis s2.Y", "r asis s2.Z", "l asis s2.X", "l asis s2.Y",
            "l asis s2.Z", "PO.X", "PO.Y", "PO.Z", "r heel s.X", "r heel s.Y",
            "r heel s.Z", "l heel s.X", "l heel s.Y", "l heel s.Z", "RHP.X", "RHP.Y",
            "RHP.Z", "LHP.X", "LHP.Y", "LHP.Z", "RK.X", "RK.Y", "RK.Z", "RTCG.X",
            "RTCG.Y", "RTCG.Z", "LK.X", "LK.Y", "LK.Z", "LTCG.X", "LTCG.Y", "LTCG.Z",
            "RA.X", "RA.Y", "RA.Z", "RCCG.X", "RCCG.Y", "RCCG.Z", "LA.X", "LA.Y",
            "LA.Z", "LCCG.X", "LCCG.Y", "LCCG.Z", "l met s.X", "l met s.Y", "l met s.Z",
            "r met s.X", "r met s.Y", "r met s.Z", "RFCG.X", "RFCG.Y", "RFCG.Z",
            "LFCG.X", "LFCG.Y", "LFCG.Z", "r should s.X", "r should s.Y",
            "r should s.Z", "l should s.X", "l should s.Y", "l should s.Z", "C7 s.X",
            "C7 s.Y", "C7 s.Z", "SHO.X", "SHO.Y", "SHO.Z"
        ]
        self.important_points = [
            "time",
            #"c7.X", "c7.Y", "c7.Z",
            "r should.X", "r should.Y", "r should.Z", "l should.X", "l should.Y", "l should.Z",
            "sacrum s.X", "sacrum s.Y", "sacrum s.Z",
            #"r asis.X", "r asis.Y", "r asis.Z", "l asis.X", "l asis.Y", "l asis.Z",
            #"MIDASIS.X", "MIDASIS.Y", "MIDASIS.Z",
            "r knee 1.X", "r knee 1.Y", "r knee 1.Z", "l knee 1.X", "l knee 1.Y", "l knee 1.Z",
            "r mall.X", "r mall.Y", "r mall.Z", "l mall.X", "l mall.Y", "l mall.Z",
            "r heel.X", "r heel.Y", "r heel.Z", "l heel.X", "l heel.Y", "l heel.Z",
            "r met.X", "r met.Y", "r met.Z", "l met.X", "l met.Y", "l met.Z",
            #"SHO.X", "SHO.Y", "SHO.Z",
            "PO.X", "PO.Y", "PO.Z"
        ]

    def load_data(self, file_path):
        """Load the data from a text file."""
        data = []
        if "force" in file_path:
            num_columns = 8
            name_columns = self.data_force_track
        if "point" in file_path:
            num_columns = 161
            name_columns = self.data_point_track
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                if len(values) == num_columns:
                    if self.remove_nan_values(values):
                        data.append(values)

        df = pd.DataFrame(data, columns=name_columns)
        df = df.apply(pd.to_numeric, errors='coerce')

        if "point" in file_path:
            df = df[self.important_points]

        return df

    def remove_nan_values(self, values):
        try:
            if math.isnan(float(values[4])) and math.isnan(float(values[7])):
                return False
        except ValueError:
            pass
        return True


class GWalk():
    def __init__(self):
        self.data_features = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                              'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw']

    def load_data(self, file_path):
        """Load the data from a text file."""
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                if len(values) == 10:
                    data.append(values)

        df = pd.DataFrame(data, columns=self.data_features)
        df = df.apply(pd.to_numeric, errors='coerce')

        return df

