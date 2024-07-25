from pandas import merge_asof

class Processing():
    def run(self, df1, df2):
        df1 = self.remove_nan(df1)
        df1 = self.interpolation(df1)

        df2 = self.remove_nan(df2)
        df2 = self.interpolation(df2)

        return self.merge(df1, df2)

    def remove_null():
        pass

    def remove_nan(self, df, param='time'):
        return df.dropna(subset=[param])

    def interpolation(self, df, param='linear'):
        return df.interpolate(method=param)

    def merge(self, df1, df2, param='time'):
        df1 = df1.sort_values(param)
        df2 = df2.sort_values(param)

        return merge_asof(df1, df2, on=param)