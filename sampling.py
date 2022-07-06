"""
The sampling module.
"""
# third party imports
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.special import kl_div #pylint: disable=no-name-in-module
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from helping_functions import get_metrics

class Sampling():
    """
    The sampling class.
    """
    def __init__(self, x__):
        self.epochs = x__.columns.values
        self.x__ = x__
        self.thresholds = np.arange(1.1, 1.5, 0.01)

    def threshold_based(self):
        """
        threshold based sampling.
        """
        results = []
        for th_h in np.arange(20,30,1):
            x_0 =  self.x__.iloc[ 0:1 ,:]
            y_df= x_0.copy()
            counter = y_df.iloc[: , 0:60].size #1st sensing interval
            for epoch in range(0, len(self.epochs) - 60 , 60):
                e_n = epoch + 60
                se_n = y_df.iloc[: , e_n:e_n + 60]
                #th_h = np.mean(se.mean() + std * se.values.std())
                #Th_L = np.mean(se.mean() - std * se.values.std())
                se_n_ = se_n[se_n > th_h]
                counter += np.count_nonzero(~np.isnan(se_n_))
                se_n_ = se_n_.ffill(axis = 'columns')
                se_n_ = se_n_.ffill()
                y_df.iloc[: , e_n:e_n + 60] = se_n_

            y_df= y_df.ffill(axis = 'columns')
            y_df= y_df.ffill()
            nmae_, rmse_ = get_metrics(x_0, y_df)
            results.append([round(th_h,1), counter, round(1- counter/2880, 2),
                            round(nmae_, 5), round(rmse_,5) ])

        results_df = pd.DataFrame(results, columns=['std', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'] )
        return results_df

    def similarity_based(self):
        """
        Similarity-based sampling.
        """
        results = []
        # Set similarity threshold
        for th_s in np.arange(0.04 , 2.2 , 0.1):
            x_0 =  self.x__.iloc[ 0:1 ,:]
            s_df = x_0.copy()
            counter = s_df.iloc[: , 0:60].size #1st sensing interval
            for epoch in range(0, len(self.epochs) - 60 , 60):
                e_n = epoch + 60
                se_ = s_df.iloc[: , epoch:e_n]
                se_n = s_df.iloc[: , e_n:e_n + 60]
                diff = abs(se_n.values - se_.values).mean()
                mode_se = mode (se_, axis=None)[0][0]
                if diff < th_s:
                    s_df.iloc[: , e_n:e_n + 60] = mode_se
                    counter += 0
                else:
                    counter += se_n.size

            nmae_, rmse_ = get_metrics(x_0, s_df)
            results.append([round(th_s, 1) ,counter,  round(1- counter/2880, 3),
                round(nmae_, 5), round(rmse_,5)] )

        df_results = pd.DataFrame(results, columns=['ThD', 'Sampled', 'Sampl%', 'NMAE', 'RMSE'])
        return df_results

    #TODO write a function to get the KL-Divergence values.
    #def get_kl()

    def voi_sampling_light(self,ref_prob, temperature_value, filename, time_dic):
        """
        voi-based sampling.
        """
        results_list = []
        #node 1 only
        x_0 =  self.x__.iloc[ 0:1 ,:]
        #X_ = round(X_0, 1)
        n_count = x_0.iloc[: , 0:60].size
        counts = []
        df_down = []
        for _ in range(len(self.thresholds)):
            counts.append(n_count)
            df_down.append(x_0.copy())
        kl_pq_l = []
        for j, _ in enumerate(self.thresholds):
            print ("The Threshold I'm working on: " , j)
            count = 0
            for epoch in range(0, len(self.epochs) - 60 , 60):
                print("The epoch I am processing is ", epoch+60)
                e_n = epoch + 60
                x_10mp = df_down[j].iloc[: , epoch:e_n]
                x_10mq = df_down[j].iloc[: , e_n:e_n + 60]
                x_10mp = round(x_10mp, 1)
                count_values = np.unique(x_10mp.values.flatten(), return_counts=True)
                prob_den_10p = pd.Series(count_values[1], index= count_values[0])
                count= prob_den_10p.combine(ref_prob, max, fill_value=0)
                p_prob= count/ count.sum()
                alphas = np.ones(len(temperature_value))
                q_prob = (alphas + count) / (count.sum() + alphas.sum())
                kl_pq = sum(kl_div(p_prob.values, q_prob.values))
                kl_pq_l.append(kl_pq)
                #ARIMA model
                x_e = x_10mp.copy()
                df_e = x_e.reset_index().melt(id_vars=['moteid'], value_vars =x_e.columns,
                        value_name='Temperature')
                df_e['time'] = df_e['epoch'].map(time_dic['time_list'])
                df_ex = df_e.set_index('time')['Temperature']

                model = ARIMA(df_ex, order=(12,1,10))
                results = model.fit()

                x_hat = results.predict(1,120)[60:]
                x_hat = round(x_hat, 1)

                if kl_pq > self.thresholds[j]:
                    counts[j] += x_10mq.size

                else:
                    df_down[j].iloc[: , e_n:e_n+60] = x_hat
                    counts[j] += 0
                count += x_10mp.size
                if epoch== len(self.epochs) - 60:
                    count += x_10mq.size
                #print("The count for this hour: ", count)
                #print("and the reduced count is: ", counts[j])
            evaluate_ = get_metrics(x_0, df_down[j])
            results_list.append([self.thresholds[j], counts[j], round(1- counts[j]/2880, 3),
                            round(evaluate_[0], 5),  round(evaluate_[1],5) ])
            print(kl_pq_l)
        df_results = pd.DataFrame(results_list, columns=['ThD', 'Sampled', 'Sampl%',
                                'NMAE', 'RMSE'])
        df_results.to_csv(filename)

    def uniform_sampling(self):
        """
        The offset is 2xsampling rate, for example when the sampling rate is the
        20 mins the offset is 40 points.
        """
        results_list = []
        downsampled_x = []
        for offset in [2, 4, 10, 40, 120, 240]:
            x_0 =  self.x__.iloc[ 0:1 ,:]
            a__ = []
            b__ = []
            for i in range(2880):
                if i %offset == 0:
                    a__.append(1)
                else:
                    a__.append(0)
            b__ = np.asarray(a__)
            m__ = b__ * x_0
            m_array = m__.replace(0, np.nan)
            x_r = m_array.ffill(axis = 'columns')
            x_mean = x_0.values.mean()
            nmae_ = mean_absolute_error(x_0, x_r, multioutput="uniform_average") / x_mean
            rmse_ = mean_squared_error(x_0,x_r, squared=False)
            print('|Sampled |Offset|Sampl%| NMAE |  RMSE |')
            print ('|' , np.count_nonzero(m__), '| ' , offset, ' |',
            round(1- np.count_nonzero(m__)/2880, 2) , '|', round(nmae_, 5),  '|', round(rmse_,5))
            results_list.append([offset, np.count_nonzero(m__),
            round(1- np.count_nonzero(m__)/2880, 2), round(nmae_, 5),  round(rmse_,5)])
            downsampled_x.append(m_array.values)
        df_results = pd.DataFrame(results_list, columns=['Offset', 'Sampled', 'Sampl%',
                            'NMAE', 'RMSE'])
        return df_results, downsampled_x
