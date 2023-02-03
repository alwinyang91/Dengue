import pandas as pd

class ProcessingData:

    def __init__(self, df, drop=True, split=True):
        self.df = df
        self.sj = None
        self.iq = None
        self.ifsplit = split
        self.ifdrop = drop

    def duplicates_drop(self,drop):
        self.ifdrop = drop
        if self.ifdrop:
            self.df.drop_duplicates(inplace=True)
            return self.df
        else:
            return self.df

    # fill missing values
    def fill_data(self, fillType):
        if fillType == 'ffill':
            return self.df.fillna(method='ffill', inplace=True)
        elif fillType == 'fmean':
            return self.df.fillna(self.df.mean(), inplace=True)
        else:
            return self.df.fillna(0, inplace=True)

    def city_split(self, splist):
        self.ifsplit = splist
        if self.ifsplit:
        # separate san juan and iquitos
            self.sj = self.df[self.df.loc[:, 'city'] == 'sj']
            self.iq = self.df[self.df.loc[:, 'city'] == 'iq']
            return self.sj, self.iq
        else:
            return self.df
