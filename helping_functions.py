"""
A module that generate all the metrics for the policy.
"""
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def get_metrics(df1_, df2_):
    """
    Results in NMAE and RMSE.
    """
    x_mean = df1_.values.mean()
    nmae_ = mean_absolute_error(df1_, df2_, multioutput="uniform_average") / x_mean
    rmse_ = mean_squared_error(df1_,df2_, squared=False)
    return (nmae_, rmse_)
