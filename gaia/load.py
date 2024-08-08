import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GaitLab:
    """Class to handle gait data from GaitLab files."""
    
    def __init__(self):
        """Initialize the GaitLab class with data column names."""
        self.data_force_track = [
            'frame', 'time',
            'r_force_x', 'r_force_y', 'r_force_z',
            'l_force_x', 'l_force_y', 'l_force_z'
        ]
        self.data_point_track = [
            "frame", "time", "c7_x", "c7_y", "c7_z", "r_should_x", "r_should_y",
            "r_should_z", "l_should_x", "l_should_y", "l_should_z", "sacrum_x",
            "sacrum_y", "sacrum_z", "r_asis_x", "r_asis_y", "r_asis_z", "r_bar 1_x",
            "r_bar 1_y", "r_bar 1_z", "r_knee 1_x", "r_knee 1_y", "r_knee 1_z",
            "r_bar 2_x", "r_bar 2_y", "r_bar 2_z", "r_mall_x", "r_mall_y", "r_mall_z",
            "r_heel_x", "r_heel_y", "r_heel_z", "r_met_x", "r_met_y", "r_met_z",
            "l_asis_x", "l_asis_y", "l_asis_z", "l_bar 1_x", "l_bar 1_y", "l_bar 1_z",
            "l_knee 1_x", "l_knee 1_y", "l_knee 1_z", "l_bar 2_x", "l_bar 2_y",
            "l_bar 2_z", "l_mall_x", "l_mall_y", "l_mall_z", "l_heel_x", "l_heel_y",
            "l_heel_z", "l_met_x", "l_met_y", "l_met_z", "l_asis s_x", "l_asis s_y",
            "l_asis s_z", "l_bar 1 s_x", "l_bar 1 s_y", "l_bar 1 s_z", "l_knee 1 s_x",
            "l_knee 1 s_y", "l_knee 1 s_z", "l_bar 2 s_x", "l_bar 2 s_y", "l_bar 2 s_z",
            "l_mall s_x", "l_mall s_y", "l_mall s_z", "sacrum s_x", "sacrum s_y",
            "sacrum s_z", "r_asis s_x", "r_asis s_y", "r_asis s_z", "r_bar 1 s_x",
            "r_bar 1 s_y", "r_bar 1 s_z", "r_knee 1 s_x", "r_knee 1 s_y",
            "r_knee 1 s_z", "r_bar 2 s_x", "r_bar 2 s_y", "r_bar 2 s_z", "r_mall s_x",
            "r_mall s_y", "r_mall s_z", "MIDASIS_x", "MIDASIS_y", "MIDASIS_z", 
            "r_asis s2_x", "r_asis s2_y", "r_asis s2_z", "l_asis s2_x", "l_asis s2_y",
            "l_asis s2_z", "PO_x", "PO_y", "PO_z", "r_heel s_x", "r_heel s_y",
            "r_heel s_z", "l_heel s_x", "l_heel s_y", "l_heel s_z", "RHP_x", "RHP_y",
            "RHP_z", "LHP_x", "LHP_y", "LHP_z", "RK_x", "RK_y", "RK_z", "RTCG_x",
            "RTCG_y", "RTCG_z", "LK_x", "LK_y", "LK_z", "LTCG_x", "LTCG_y", "LTCG_z",
            "RA_x", "RA_y", "RA_z", "RCCG_x", "RCCG_y", "RCCG_z", "LA_x", "LA_y",
            "LA_z", "LCCG_x", "LCCG_y", "LCCG_z", "l_met s_x", "l_met s_y", "l_met s_z",
            "r_met s_x", "r_met s_y", "r_met s_z", "RFCG_x", "RFCG_y", "RFCG_z",
            "LFCG_x", "LFCG_y", "LFCG_z", "r_should s_x", "r_should s_y",
            "r_should s_z", "l_should s_x", "l_should s_y", "l_should s_z", "C7 s_x",
            "C7 s_y", "C7 s_z", "SHO_x", "SHO_y", "SHO_z"
        ]
        self.important_points = [
            "time",
            "c7_x", "c7_y", "c7_z",
            "r_should_x", "r_should_y", "r_should_z", "l_should_x", "l_should_y", "l_should_z",
            "sacrum s_x", "sacrum s_y", "sacrum s_z",
            "r_asis_x", "r_asis_y", "r_asis_z", "l_asis_x", "l_asis_y", "l_asis_z",
            "MIDASIS_x", "MIDASIS_y", "MIDASIS_z",
            "r_knee 1_x", "r_knee 1_y", "r_knee 1_z", "l_knee 1_x", "l_knee 1_y", "l_knee 1_z",
            "r_mall_x", "r_mall_y", "r_mall_z", "l_mall_x", "l_mall_y", "l_mall_z",
            "r_heel_x", "r_heel_y", "r_heel_z", "l_heel_x", "l_heel_y", "l_heel_z",
            "r_met_x", "r_met_y", "r_met_z", "l_met_x", "l_met_y", "l_met_z",
            "SHO_x", "SHO_y", "SHO_z",
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
