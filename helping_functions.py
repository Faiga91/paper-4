"""
A module that generate all the metrics for the policy.
"""
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch

def get_metrics(df1_, df2_):
    """
    Results in NMAE and RMSE.
    """
    x_mean = df1_.values.mean()
    nmae_ = mean_absolute_error(df1_, df2_, multioutput="uniform_average") / x_mean
    rmse_ = mean_squared_error(df1_,df2_, squared=False)
    return ([nmae_, rmse_])

def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = torch.min(data)
    data = data - min_val
    max_val = torch.max(data)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val