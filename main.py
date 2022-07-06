"""
Main module that run all the other submodules in this folder.
"""
#%%
# standard imports
import warnings

# third party import
import pandas as pd
import numpy as np

from get_data import Data
import plot_graphs
from sampling import Sampling
#%%
warnings.filterwarnings("ignore")
#%%
mydata = Data()

X_day_temp = mydata.get_day()
#%%

# Node 18 is an outlie
#plot_graphs.plot_node_Temperature(X_day_temp, 18)
#X_day_temp_no = remove_outliers(X_day_temp, [8,18, 50, 53])
X = mydata.get_array_x(X_day_temp[['epoch', 'moteid', 'Temperature']])
X_Day = X_day_temp[['epoch', 'moteid', 'Temperature']]

X_day_temp['time_list'] = pd.to_datetime(X_day_temp['date'] + ' ' + X_day_temp['time'])
time_dic = X_day_temp[['time_list','epoch']]
time_dic = time_dic.drop_duplicates(subset=['epoch'])
time_dic = time_dic.set_index('epoch')
# %%

down_sampling = Sampling(X)

uniform_samp_results , downsampled_X = down_sampling.uniform_sampling()
print(uniform_samp_results)

plot_graphs.plot_node_Temperature(X_day_temp, 1.0)

#%%
#print('Spatially Uniform Sampling')
#sampling.spatially_uniform(X)


print("Threshold method")
thr_results_df = down_sampling.threshold_based()
print(thr_results_df)

# Based on data similarity
print("Similarity-based methods")
sim_results_df = down_sampling.similarity_based()
print(sim_results_df)


# Based on data similarity
#sim_results_df_arima = down_sampling.similarity_based_ARIMA(time_dic)
#print(sim_results_df_arima)

#%%

X_day_temp['Temperature'] =  X_day_temp['Temperature'].apply(lambda x : np.ceil(10 * x) / 10)

Temp_probabilities = X_day_temp.groupby('Temperature').size().div(len(X_day_temp))
ref_prob = pd.Series(1e-10, index=Temp_probabilities.index)
Temperature_value = ref_prob.index.values.tolist()

down_sampling.voi_sampling_light(ref_prob, Temperature_value,
    './Results/VOI_results.csv', time_dic)
df_results_VoI = pd.read_csv("./Results/VOI_results.csv")
#%%
sim_results_df['ThD'] = sim_results_df['ThD'].round(2)
df_results_VoI['ThD'] = df_results_VoI['ThD'].round(1)

results_plots = plot_graphs.plot_results(thr_results_df, sim_results_df[::2], df_results_VoI[::4] )
results_plots.plot_results1('./Figures/results1.pdf')
results_plots.plot_results2('./Figures/results2.pdf')
results_plots.plot_results3('./Figures/results3.pdf')
# %%
# %%
