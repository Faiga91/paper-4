"""
Module for reading the data into dataframes from the .csv data file.
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore

class Data():
    """
    data class with 5 functions: clean_data, remove_outliers, get_day, get_array_X, and get_week.
    """
    def __init__(self):
        self.data_path = './Data/data.csv'
        self.data = pd.read_csv(self.data_path, names=['date' , 'time' , 'epoch', 'moteid',
                        'Temperature', 'Humidity', 'Light', 'Voltage'])
        self.data_clean = self.remove_outliers()

    def remove_outliers(self):
        """
        Remove nodes with outliers from the data. Those nodes with ID: 5,8,15,18,50,53.
        """
        nodes = np.array([x for x in np.arange(1,55,1) if x not in [5,8,15,18,50,53]])
        data_clean = self.data[self.data["moteid"].isin(nodes)]
        data_clean = data_clean.dropna()
        data_clean["Temperature"] =  data_clean["Temperature"].clip(upper = 37)
        data_clean['timestamp'] = pd.to_datetime (data_clean['date'] + ' ' + data_clean['time'])
        data_clean = data_clean.set_index('timestamp')
        return data_clean

    def clean_data(self, df_):
        """
        Remove outliers using the Z-score.
        """
        z_scores = zscore(df_)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        return filtered_entries


    def get_day(self):
        """
        This function will read the data from the specified file.
        Then it will choose one day from the dataset that has the least missing data points.
        """
        # Determine the number of missing data points in each day of the data
        missing_p = []
        dates = []

        for date_x in self.data_clean['date'].unique():
            #Percentage of missing data in the date
            p_missing = len(self.data_clean[self.data_clean['date'] == date_x]) / (2880 * 54)
            p_missing = 1 - p_missing
            dates.append(date_x)
            missing_p.append(round(p_missing * 100 ,2))

        df_missing_p = pd.DataFrame()
        df_missing_p['dates'] = dates
        df_missing_p['missing%'] = missing_p

        least_missing = df_missing_p[df_missing_p['missing%'] == df_missing_p['missing%'].min()]
        print("The day with the least missing values:" , least_missing.values[0][0])

        # Choose the day with the least missing values
        x_day = self.data_clean[self.data_clean['date'] == least_missing.values[0][0]]
        x_day_temp = x_day[['time', 'epoch', 'moteid', 'Temperature', 'date']]

        return x_day_temp

    def get_array_x(self, df_):
        """
        Fill the missing data and use pivot to get the data in an array shape.
        """
        x_array = df_.pivot(index='moteid', columns = 'epoch', values='Temperature')
        # Fill missing data using forward fill => if the node did not make
        # a new measurement it will buffer the previous one
        x_array = x_array.ffill(axis = 'columns')
        x_array = x_array.bfill(axis = 'columns')
        return x_array

    def get_week(self):
        """
        Choose the first week of March as training set.
        """
        self.data_clean = self.remove_outliers()
        week = self.data_clean[self.data_clean['date'].between('2004-03-01','2004-03-07')]
        return week


def get_mote_locations(path):
    """
    Read the nodes location (x,y) to a dataframe.
    """
    # path = './Data/mote_locs.csv'
    mote_locs = pd.read_csv( path, names=['moteid', 'x_loc' , 'y_loc'])
    return mote_locs

def get_connectivity_info (path):
    """
    Read the connectivity success probability into a dataframe.
    """
    # path = './Data/connectivity.csv'
    connectivity = pd.read_csv( path, names=['sender_id', 'receiver_id', 'trans_success_rate'])
    return connectivity

def clusters(df_):
    """
    Divide the nodes into clusters.
    """
    c1_ = [x for x in range(1,11) if x not in [5,8]]
    c2_ = [x for x in range(11,21) if x not in [15,18]]
    c3_ = list(range(21,29))
    c4_ = list(range(29,37))
    c5_ = list(range(37,45))
    c6_ = [x for x in range(45,55) if x not in [50, 53]]

    df1 = df_[df_['moteid'].isin(c1_)]
    df2 = df_[df_['moteid'].isin(c2_)]
    df3 = df_[df_['moteid'].isin(c3_)]
    df4 = df_[df_['moteid'].isin(c4_)]
    df5 = df_[df_['moteid'].isin(c5_)]
    df6 = df_[df_['moteid'].isin(c6_)]

    return (df1, df2, df3, df4, df5, df6)

def light_level(light):
    """
    Classify the light level into 5 classes: 'Dark', 'Little Dark', 'Bright', and 'Very Bright'.
    """
    if light < 50:
        return 'Dark'
    if 50 <= light < 100 :
        return 'Little Dark'
    if 100 <= light < 200 :
        return 'Little Bright'
    if 200 <= light < 500:
        return 'Bright'
    if light > 500:
        return 'Very Bright'
    return None
