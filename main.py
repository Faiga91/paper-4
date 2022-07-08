"""
Main module that run all the other submodules in this folder.
"""
#%%
# standard imports
import warnings

# third party import
import pandas as pd
from IPython import get_ipython

from get_data import Data, get_array_x
import plot_graphs
from sampling import Sampling

# Auto load changes in other referenced packages
IPYTHON_INSTANCE = get_ipython()
IPYTHON_INSTANCE.run_line_magic('load_ext','autoreload')
IPYTHON_INSTANCE.run_line_magic('autoreload','2')
#%%
warnings.filterwarnings("ignore")

mydata = Data()
X_day_temp = mydata.get_day()
#%%

# Node 18 is an outlie
#plot_graphs.plot_node_temperature(X_day_temp, 18)
#X_day_temp_no = remove_outliers(X_day_temp, [8,18, 50, 53])
X = get_array_x(X_day_temp[['epoch', 'moteid', 'Temperature']])
X_Day = X_day_temp[['epoch', 'moteid', 'Temperature']]

X_day_temp['time_list'] = pd.to_datetime(X_day_temp['date'] + ' ' + X_day_temp['time'])
time_dic = X_day_temp[['time_list','epoch']]
time_dic = time_dic.drop_duplicates(subset=['epoch'])
time_dic = time_dic.set_index('epoch')
# %%

down_sampling = Sampling(X, time_dic)

uniform_samp_results , downsampled_X = down_sampling.uniform_sampling()
print(uniform_samp_results)

plot_graphs.plot_node_temperature(X_day_temp, 1.0)

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


#%%
sim_results_df['ThD'] = sim_results_df['ThD'].round(2)

results_plots = plot_graphs.PlotResults(thr_results_df, sim_results_df[::2])
results_plots.plot_results1('./Figures/results1.pdf')
results_plots.plot_results2('./Figures/results2.pdf')
results_plots.plot_results3('./Figures/results3.pdf')

# %%
