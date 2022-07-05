import pandas as pd 
import numpy as np
from scipy.stats import zscore

class data():
    def __init__(self):
        self.data_path = './Data/data.csv'
        self.data = pd.read_csv(self.data_path, names=['date' , 'time' , 'epoch', 'moteid', 
                        'Temperature', 'Humidity', 'Light', 'Voltage'])
        self.data_clean = self.remove_outliers()

    def remove_outliers(self):
        nodes = np.array([x for x in np.arange(1,55,1) if x not in [5,8,15,18,50,53]]) 
        data_clean = self.data[self.data["moteid"].isin(nodes)]
        data_clean = data_clean.dropna()
        data_clean["Temperature"] =  data_clean["Temperature"].clip(upper = 37)
        data_clean['timestamp'] = pd.to_datetime (data_clean['date'] + ' ' + data_clean['time'])
        data_clean = data_clean.set_index('timestamp')
        return data_clean

    def clean_data(self, df):
        z_scores = zscore(df)
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
            p = len(self.data_clean[self.data_clean['date'] == date_x]) / (2880 * 54)
            p = 1 - p
            dates.append(date_x)
            missing_p.append(round(p * 100 ,2))

        df_missing_p = pd.DataFrame()
        df_missing_p['dates'] = dates 
        df_missing_p['missing%'] = missing_p 

        least_missing = df_missing_p[df_missing_p['missing%'] == df_missing_p['missing%'].min()]
        print("The day with the least missing values:" , least_missing.values[0][0])

        # Choose the day with the least missing values 
        X_day = self.data_clean[self.data_clean['date'] == least_missing.values[0][0]]
        X_day_temp = X_day[['time', 'epoch', 'moteid', 'Temperature', 'date']]

        return X_day_temp

    def get_array_X(self, df):
        X = df.pivot(index='moteid', columns = 'epoch', values='Temperature')
        # Fill missing data using forward fill => if the node did not make
        # a new measurement it will buffer the previous one
        X = X.ffill(axis = 'columns')
        X = X.bfill(axis = 'columns')
        return X

    def get_week(self):
        # Choose the first week of March as training set 
        self.data_clean = self.remove_outliers()
        week = self.data_clean[self.data_clean['date'].between('2004-03-01','2004-03-07')]
        return week
    

def get_mote_locations(path):
    # path = './Data/mote_locs.csv'
    mote_locs = pd.read_csv( path, names=['moteid', 'x_loc' , 'y_loc'])
    return mote_locs

def get_connectivity_info (path): 
    # path = './Data/connectivity.csv'
    connectivity = pd.read_csv( path, names=['sender_id', 'receiver_id', 'trans_success_rate'])
    return connectivity

def clusters(df):
    C1 = [x for x in range(1,11) if x not in [5,8]]
    C2 = [x for x in range(11,21) if x not in [15,18]]
    C3 = [x for x in range(21,29)]
    C4 = [x for x in range(29,37)]
    C5 = [x for x in range(37,45)]
    C6 = [x for x in range(45,55) if x not in [50, 53]]

    df1 = df[df['moteid'].isin(C1)]
    df2 = df[df['moteid'].isin(C2)]
    df3 = df[df['moteid'].isin(C3)]
    df4 = df[df['moteid'].isin(C4)]
    df5 = df[df['moteid'].isin(C5)]
    df6 = df[df['moteid'].isin(C6)]

    return (df1, df2, df3, df4, df5. df6)

def light_level(light):
    if light < 50:
        return 'Dark'
    elif 50 <= light < 100 :
        return 'Little Dark'
    elif 100 <= light < 200 :
        return 'Little Bright'
    elif 200 <= light < 500:
        return 'Bright'
    elif light > 500:
        return 'Very Bright'