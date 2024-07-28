from .filtering import Filtering
from pandas import DataFrame, merge_asof
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Processing():
    def __init__(self):
        self.filtering = Filtering()

    def run(self, df1, df2):
        df1 = self.remove_nan(df1)
        df1 = self.convert_nan(df1)
        df1 = Filtering.butter_lowpass(df1)
        #df1 = self.interpolation(df1)
        #df1 = self.normalize_data(df1, scaler_type='standard')
        #df1 = self.normalize_data(df1, scaler_type='minmax')

        df2 = self.remove_nan(df2)
        df2 = self.convert_nan(df2)
        df2 = Filtering.butter_lowpass(df2)
        #df2 = self.interpolation(df2)
        #df2 = self.normalize_data(df2, scaler_type='standard')
        #df2 = self.normalize_data(df2, scaler_type='minmax')

        return self.merge(df1, df2)

    def remove_nan(self, df, param='time'):
        return df.dropna(subset=[param])
    
    def convert_nan(self, df, type="mean"):
        if type == "mean":
            df.fillna(df.mean(), inplace=True)
        elif type == "zero":
            df.fillna(0)
        return df

    def normalize_data(self, df, scaler_type='standard'):
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        scaled_data = scaler.fit_transform(df.drop(columns=['time']))
        df_scaled = DataFrame(scaled_data, columns=df.columns.drop('time'))
        df_scaled['time'] = df['time'].values
        return df_scaled

    def interpolation(self, df, param='linear'):
        return df.interpolate(method=param)

    def merge(self, df1, df2, param='time'):
        df1 = df1.sort_values(param)
        df2 = df2.sort_values(param)

        return merge_asof(df1, df2, on=param)