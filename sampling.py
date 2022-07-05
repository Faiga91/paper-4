import numpy as np
from scipy.special import rel_entr, kl_div
from get_data import data
from helping_functions import get_metrics
import pandas as pd 
from scipy.stats import mode
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class sampling():
    def __init__(self, X):
        self.epochs = X.columns.values
        self.X = X

    def threshold_based(self): 
        results = []
        for Th_H in np.arange(20,30,1):
            X_0 =  self.X.iloc[ 0:1 ,:]
            Y = X_0.copy()
            counter = Y.iloc[: , 0:60].size #1st sensing interval
            for e in range(0, len(self.epochs) - 60 , 60):
                e_n = e + 60
                se = Y.iloc[: , e:e_n]
                se_n = Y.iloc[: , e_n:e_n + 60]
                #Th_H = np.mean(se.mean() + std * se.values.std())
                #Th_L = np.mean(se.mean() - std * se.values.std())
                se_n_ = se_n[se_n > Th_H]
                counter += np.count_nonzero(~np.isnan(se_n_))
                se_n_ = se_n_.ffill(axis = 'columns')     
                se_n_ = se_n_.ffill()
                Y.iloc[: , e_n:e_n + 60] = se_n_
                
            Y = Y.ffill(axis = 'columns')   
            Y = Y.ffill()  
            NMAE, RMSE = get_metrics(X_0, Y)
            results.append([round(Th_H,1), counter, round(1- counter/2880, 2), 
                            round(NMAE, 5), round(RMSE,5) ])
            
        results_df = pd.DataFrame(results, columns=['std', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'] )
        return(results_df)

    def similarity_based(self):
        results = []
        # Set similarity threshold
        for TH_s in np.arange(0.04 , 2.2 , 0.1):
            X_0 =  self.X.iloc[ 0:1 ,:]
            S = X_0.copy() 
            counter = S.iloc[: , 0:60].size #1st sensing interval
            for e in range(0, len(self.epochs) - 60 , 60):
                e_n = e + 60
                se = S.iloc[: , e:e_n]
                se_n = S.iloc[: , e_n:e_n + 60]
                diff = abs(se_n.values - se.values).mean() 
                mode_se = mode (se, axis=None)[0][0]
                if diff < TH_s:
                    S.iloc[: , e_n:e_n + 60] = mode_se
                    counter += 0
                else: 
                    counter += se_n.size

            NMAE, RMSE = get_metrics(X_0, S)
            results.append([round(TH_s, 1) ,counter,  round(1- counter/2880, 3), round(NMAE, 5), round(RMSE,5)] )

        df_results = pd.DataFrame(results, columns=['ThD', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'])
        return df_results

    def voi_sampling_light(self,ref_prob, Temperature_value, filename, Thresholds, time_dic):
        results_list = []
        #node 1 only
        X_0 =  self.X.iloc[ 0:1 ,:]
        #X_ = round(X_0, 1)
        N_count = X_0.iloc[: , 0:60].size
        Counts = []
        df_down = []
        for m in range(len(Thresholds)):
            Counts.append(N_count)
            df_down.append(X_0.copy())
        kl_pq_l = []
        for j in range(len(Thresholds)):
            print ("The Threshold I'm working on: " , j)
            count = 0
            for e in range(0, len(self.epochs) - 60 , 60):
                print("The epoch I am processing is ", e+60)
                e_n = e + 60
                X_10mp = df_down[j].iloc[: , e:e_n]
                X_10mq = df_down[j].iloc[: , e_n:e_n + 60]
                X_10mp_round = round(X_10mp, 1)
                count_values = np.unique(X_10mp_round.values.flatten(), return_counts=True)
                prob_den_10p = pd.Series(count_values[1], index= count_values[0]) 
                C = prob_den_10p.combine(ref_prob, max, fill_value=0) 
                P = C / C.sum()
                #P_mode = mode(X_10mp, axis=None)[0][0]
                alphas = np.ones(len(Temperature_value))
                Q = (alphas + C) / (C.sum() + alphas.sum())
                kl_pq = sum(kl_div(P.values, Q.values))
                kl_pq_l.append(kl_pq)

                #ARIMA model 
                X_e = X_10mp.copy()
                df_e = X_e.reset_index().melt(id_vars=['moteid'], value_vars =X_e.columns, value_name='Temperature')
                df_e['time'] = df_e['epoch'].map(time_dic['time_list'])
                df_ex = df_e.set_index('time')['Temperature']

                decomposition = seasonal_decompose(df_ex, period = 1) 
                model = ARIMA(df_ex, order=(12,1,10))
                results = model.fit()

                forecast = results.get_prediction(1, 120)
                mean_forecast = forecast.predicted_mean 
                confidence_intervals = forecast.conf_int()  
                x_hat = results.predict(1,120)[60:]
                x_hat = round(x_hat, 1)

                if kl_pq > Thresholds[j]: 
                    Counts[j] += X_10mq.size

                else: 
                    df_down[j].iloc[: , e_n:e_n+60] = x_hat
                    Counts[j] += 0
                count += X_10mp.size
                if e == len(self.epochs) - 60:
                    count += X_10mq.size
                #print("The count for this hour: ", count)
                #print("and the reduced count is: ", Counts[j])
            NMAE, RMSE = get_metrics(X_0, df_down[j])
            results_list.append([Thresholds[j], Counts[j], round(1- Counts[j]/2880, 3), 
                            round(NMAE, 5),  round(RMSE,5) ])
            print(kl_pq_l)
        df_results = pd.DataFrame(results_list, columns=['ThD', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'])
        df_results.to_csv(filename)

    def similarity_based_ARIMA(self, time_dic):
        results_list = []
        # Set similarity threshold
        for TH_s in np.arange(0.004, 0.025,0.0005):
            X_0 =  self.X.iloc[ 0:1 ,:]
            S = X_0.copy() 
            counter = S.iloc[: , 0:60].size #1st sensing interval
            for e in range(0, len(self.epochs) - 60 , 60):
                e_n = e + 60
                se = S.iloc[: , e:e_n]

                #ARIMA model 
                X_e = se.copy()
                df_e = X_e.reset_index().melt(id_vars=['moteid'], value_vars =X_e.columns, value_name='Temperature')
                df_e['time'] = df_e['epoch'].map(time_dic['time_list'])
                df_ex = df_e.set_index('time')['Temperature']

                decomposition = seasonal_decompose(df_ex, period = 1) 
                model = ARIMA(df_ex, order=(2,1,2))
                results = model.fit()

                forecast = results.get_prediction(1, 120)
                mean_forecast = forecast.predicted_mean 
                confidence_intervals = forecast.conf_int()  
                x_hat = results.predict(1,120)[60:]
                x_hat = round(x_hat, 1)

                se_n = x_hat.copy()
                diff = abs(se_n.values - se.values).mean() 
                mode_se = mode (se, axis=None)[0][0]

                if diff < TH_s:
                    S.iloc[: , e_n:e_n + 60] = x_hat
                    counter += 0
                else: 
                    counter += se_n.size

            NMAE, RMSE = get_metrics(X_0, S)
            results_list.append([round(TH_s, 5) ,counter,  round(1- counter/2880, 3), round(NMAE, 5), round(RMSE,5)] )

        df_results = pd.DataFrame(results_list, columns=['ThD', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'])
        return df_results

    def uniform_sampling(self):
        """
        The offset is 2xsampling rate, for example when the sampling rate is the
        20 mins the offset is 40 points. 
        """
        results_list = []
        downsampled_X = []
        for offset in [2, 4, 10, 40, 120, 240]:
            X_0 =  self.X.iloc[ 0:1 ,:]
            a = []
            A = []
            for i in range(2880):
                if (i %offset == 0):
                    a.append(1)
                else:
                    a.append(0)
            A = np.asarray(a)
            M = A * X_0
            M_ = M.replace(0, np.nan)
            X_r = M_.ffill(axis = 'columns')
            X_mean = X_0.values.mean()
            NMAE = mean_absolute_error(X_0, X_r, multioutput="uniform_average") / X_mean 
            RMSE = mean_squared_error(X_0,X_r, squared=False) 
            print('|Sampled |Offset|Sampl%| NMAE |  RMSE |')
            print ('|' , np.count_nonzero(M), '| ' , offset, ' |', 
            round(1- np.count_nonzero(M)/2880, 2) , '|', round(NMAE, 5),  '|', round(RMSE,5))
            results_list.append([offset, np.count_nonzero(M), round(1- np.count_nonzero(M)/2880, 2) , 
                       round(NMAE, 5),  round(RMSE,5)])
            downsampled_X.append(M_.values)
        df_results = pd.DataFrame(results_list, columns=['Offset', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'])
        return df_results, downsampled_X