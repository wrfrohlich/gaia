from pandas import merge_asof

class Processing():
    def run(self, df1, df2):
        self.remove_nan(df1)
        self.interpolation(df1)

        self.remove_nan(df2)
        self.interpolation(df2)

        self.merge(df1, df2)

    def remove_null():
        pass

    def remove_nan(self, df, param='time'):
        df.dropna(subset=[param])
        return df

    def interpolation(self, df, param='linear'):
        df.interpolate(method=param)
        return df
    
    def merge(self, df1, df2, param='time'):
        df1 = df1.sort_values(param)
        df2 = df2.sort_values(param)

        return merge_asof(df1, df2, on=param)