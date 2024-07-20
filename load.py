"""
Load Gait files
"""

import math

class GaitLab():
    def load_data(self, file_path):
        """Load the data from a text file."""
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                if len(values) == 8:
                    if self.remove_nan_values(values):
                        data.append(values)
                        print(values)
        return data

    def remove_nan_values(self, values):
        try:
            if math.isnan(float(values[4])) and math.isnan(float(values[7])):
                return False
        except ValueError:
            pass
        return True


class GWalk():
    def load_data(self, file_path):
        """Load the data from a text file."""
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                data.append(values)
