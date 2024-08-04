import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GaitLab:
    """Class to handle gait data from GaitLab files."""
    
    def __init__(self):
        """Initialize the GaitLab class with data column names."""
        self.data_force_track = [
            'frame', 'time', 'rx', 'ry', 'rz', 'lx', 'ly', 'lz'
        ]
        self.data_point_track = [
            "frame", "time", "c7_x", "c7_y", "c7_z", "r should_x", "r should_y",
            "r should_z", "l should_x", "l should_y", "l should_z", "sacrum_x",
            "sacrum_y", "sacrum_z", "r asis_x", "r asis_y", "r asis_z", "r bar 1_x",
            "r bar 1_y", "r bar 1_z", "r knee 1_x", "r knee 1_y", "r knee 1_z",
            "r bar 2_x", "r bar 2_y", "r bar 2_z", "r mall_x", "r mall_y", "r mall_z",
            "r heel_x", "r heel_y", "r heel_z", "r met_x", "r met_y", "r met_z",
            "l asis_x", "l asis_y", "l asis_z", "l bar 1_x", "l bar 1_y", "l bar 1_z",
            "l knee 1_x", "l knee 1_y", "l knee 1_z", "l bar 2_x", "l bar 2_y",
            "l bar 2_z", "l mall_x", "l mall_y", "l mall_z", "l heel_x", "l heel_y",
            "l heel_z", "l met_x", "l met_y", "l met_z", "l asis s_x", "l asis s_y",
            "l asis s_z", "l bar 1 s_x", "l bar 1 s_y", "l bar 1 s_z", "l knee 1 s_x",
            "l knee 1 s_y", "l knee 1 s_z", "l bar 2 s_x", "l bar 2 s_y", "l bar 2 s_z",
            "l mall s_x", "l mall s_y", "l mall s_z", "sacrum s_x", "sacrum s_y",
            "sacrum s_z", "r asis s_x", "r asis s_y", "r asis s_z", "r bar 1 s_x",
            "r bar 1 s_y", "r bar 1 s_z", "r knee 1 s_x", "r knee 1 s_y",
            "r knee 1 s_z", "r bar 2 s_x", "r bar 2 s_y", "r bar 2 s_z", "r mall s_x",
            "r mall s_y", "r mall s_z", "MIDASIS_x", "MIDASIS_y", "MIDASIS_z", 
            "r asis s2_x", "r asis s2_y", "r asis s2_z", "l asis s2_x", "l asis s2_y",
            "l asis s2_z", "PO_x", "PO_y", "PO_z", "r heel s_x", "r heel s_y",
            "r heel s_z", "l heel s_x", "l heel s_y", "l heel s_z", "RHP_x", "RHP_y",
            "RHP_z", "LHP_x", "LHP_y", "LHP_z", "RK_x", "RK_y", "RK_z", "RTCG_x",
            "RTCG_y", "RTCG_z", "LK_x", "LK_y", "LK_z", "LTCG_x", "LTCG_y", "LTCG_z",
            "RA_x", "RA_y", "RA_z", "RCCG_x", "RCCG_y", "RCCG_z", "LA_x", "LA_y",
            "LA_z", "LCCG_x", "LCCG_y", "LCCG_z", "l met s_x", "l met s_y", "l met s_z",
            "r met s_x", "r met s_y", "r met s_z", "RFCG_x", "RFCG_y", "RFCG_z",
            "LFCG_x", "LFCG_y", "LFCG_z", "r should s_x", "r should s_y",
            "r should s_z", "l should s_x", "l should s_y", "l should s_z", "C7 s_x",
            "C7 s_y", "C7 s_z", "SHO_x", "SHO_y", "SHO_z"
        ]
        self.important_points = [
            "time",
            #"c7_x", "c7_y", "c7_z",
            "r should_x", "r should_y", "r should_z", "l should_x", "l should_y", "l should_z",
            "sacrum s_x", "sacrum s_y", "sacrum s_z",
            #"r asis_x", "r asis_y", "r asis_z", "l asis_x", "l asis_y", "l asis_z",
            #"MIDASIS_x", "MIDASIS_y", "MIDASIS_z",
            "r knee 1_x", "r knee 1_y", "r knee 1_z", "l knee 1_x", "l knee 1_y", "l knee 1_z",
            "r mall_x", "r mall_y", "r mall_z", "l mall_x", "l mall_y", "l mall_z",
            "r heel_x", "r heel_y", "r heel_z", "l heel_x", "l heel_y", "l heel_z",
            "r met_x", "r met_y", "r met_z", "l met_x", "l met_y", "l met_z",
            #"SHO_x", "SHO_y", "SHO_z",
            "PO_x", "PO_y", "PO_z"
        ]

    def load_data(self, file_path):
        """Load gait data from a text file.

        Args:
            file_path (str): The path to the data file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        data = []
        num_columns, name_columns = self._get_file_parameters(file_path)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                if len(values) == num_columns and self._is_valid_record(values):
                    data.append(values)

        df = pd.DataFrame(data, columns=name_columns)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Uncomment this if you need to filter columns based on `important_points`
        # if "point" in file_path:
        #     df = df[self.important_points]

        return df

    def _get_file_parameters(self, file_path):
        """Determine the number of columns and column names based on file type.

        Args:
            file_path (str): The path to the data file.

        Returns:
            tuple: Number of columns and column names.
        """
        if "force" in file_path:
            return 8, self.data_force_track
        elif "point" in file_path:
            return 161, self.data_point_track
        else:
            raise ValueError("Unsupported file type")

    def _is_valid_record(self, values):
        """Check if a record has valid data, excluding NaN values.

        Args:
            values (list): List of values from a record.

        Returns:
            bool: True if record is valid, False otherwise.
        """
        try:
            if math.isnan(float(values[4])) and math.isnan(float(values[7])):
                return False
        except ValueError:
            pass
        return True


class GWalk:
    """Class to handle gait data from GWalk files."""
    
    def __init__(self):
        """Initialize the GWalk class with data column names."""
        self.data_features = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                              'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw']

    def load_data(self, file_path):
        """Load gait data from a text file.

        Args:
            file_path (str): The path to the data file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
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
