"""
Module for all the plots related to this repository.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter

def set_plot_env():
    """
    Setup function.
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=2)
    fig_, ax_ = plt.subplots(figsize=(10, 5))
    return fig_, ax_

def set_heatmap_env():
    """
    Plot heatmap.
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=1.5)
    fig_, ax_ = plt.subplots(figsize=(10, 10))
    return fig_, ax_

def plot_node_temperature(df_, node_id):
    """
    Plotting the temperature.
    Inputs:
    - df_: the dataframe containing the temp values.
    - node_id: the node id, int between 1-54.
    """
    _, axes  = plt.subplots(1,4,figsize=(20, 7))

    df_ = df_[df_['moteid'] == node_id]
    df_ = df_.set_index('time_list')

    sns.scatterplot(x = "epoch", y = "Temperature", data = df_, label='Original',
                    marker='X', ax = axes[0])
    sns.scatterplot(x = "epoch", y="Temperature", data =df_[::2], label ='50%',
                color='green', marker='X', ax=axes[1])
    sns.scatterplot(x = "epoch", y="Temperature", data =df_[::4], label ='75%',
                color='red', marker='X', ax=axes[2])
    sns.scatterplot(x = "epoch", y="Temperature", data =df_[::10], label ='90%',
            color='k', marker='X', ax=axes[3])

    plt.ylabel('Temperature')

    for ax_ in [axes[1], axes[2], axes[3]]:
        ax_.set(ylabel=None)
        ax_.set(yticklabels=[])

    plt.savefig('./Figures/downsampling.pdf' , bbox_inches='tight')
    plt.show()

def plot_temperature(df_, file):
    """
    plot the temperature variable from all the nodes
    """
    _, _ = set_plot_env()
    sns.lineplot(x = df_['epoch'] , y = df_['Temperature'], color = 'purple')
    plt.savefig(file , bbox_inches='tight')
    plt.show()

def heatmap_hour_temperature(x__):
    """
    Plot heatmap for only one hour.
    Input:
    X: x-array containing the temp values.
    """
    #Show the heat map for that  hour
    colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']
    hour_start = np.arange(25804,28683 + 120 ,120)
    for i in range(len(hour_start) -1):
        _, _ = set_heatmap_env()
        hour_v = np.arange(hour_start[i],hour_start[i+1])
        x_hour = x__[hour_v]
        sns.heatmap(x_hour, yticklabels=True, vmin = 18, vmax = 38, cmap=sns.color_palette(colors))
        plt.savefig('./Figures/' + 'x_hour' +str(i), bbox_inches='tight')

def plot_temperature_array(x__):
    """
    plot the temperature for each mote in the array X
    """
    for mote in x__.index:
        x_mote = x__.loc[mote]
        _, _ = set_plot_env()
        sns.scatterplot(x = x_mote.index, y = x_mote)
        plt.savefig('./Figures/X_mote/' + 'X_mote' + str(mote) + '.png', bbox_inches='tight')

def plot_density(df_, file):
    """
    Plot the density and save it into a fig file.
    Input:
    df_ - the dataframe.
    file - where to save.
    """
    _, _ = set_heatmap_env()
    df_.plot()
    plt.savefig(file, bbox_inches='tight')

def show_result(y_test, predicted):
    """
    Plot the prediction Vs. the real values.
    """
    _, _ = set_plot_env()
    plt.plot(y_test.index, predicted, 'o-', label="predicted")
    plt.plot(y_test.index, y_test, '.-', label="actual")
    plt.ylabel("Temperature")
    plt.legend()

def show_results_with_ci(x_test, prediction_mean, y_test, ci_, filename):
    """
    Plot the results with confidence interval.
    ci = a list with both the upper and lower values for the ci_.
    ci_[0] = lower, and ci_[1] = upper.
    """
    _, ax_ = set_plot_env()
    plt.plot(x_test, prediction_mean, color='red', label="predicted mean")
    plt.plot(x_test, y_test, '.-', label="actual")
    plt.fill_between(x_test, ci_[1], ci_[0], color='gray', label="95% CI")
    plt.ylabel("Temperature")
    ax_.set_xticklabels(ax_.get_xticklabels(),rotation = 30)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def theme(subplots_rows, subplots_col):
    """
    Set up method.
    """
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1.5)
    fig_, ax_ = plt.subplots(subplots_rows,subplots_col,figsize=(12, 6))
    return fig_, ax_

class PlotResults():
    """
    Class to plot the results.
    """
    def __init__(self, threshold_results, similarity_results):
        self.thr_res = threshold_results
        self.sim_res = similarity_results

    def plot_results1(self, filename):
        """
        Plot the results with lines for each algorithm - use RMSE.
        """
        _, _ = theme(1,1)

        sns.lineplot(x = self.thr_res['Sampled'], y = self.thr_res['RMSE'] , color='#d95f02',
                            label='Threshold-based Scheduling')
        sns.scatterplot(x = 'Sampled', y = 'RMSE' , data= self.thr_res,
                        color='#d95f02', marker = 'o', s=50)

        sns.lineplot(x ='Sampled', y= 'RMSE' , data= self.sim_res,
                    color='#1b9e77', linestyle='--',
                        label='Similarity-based Scheduling')
        sns.scatterplot(x ='Sampled', y='RMSE' , data= self.sim_res,
                    color='#1b9e77', marker = 'o', s=50)

        plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def plot_results2(self, filename):
        """
        Plot the results with line for each alogorithm - use NMAE.
        """
        _, ax_ = theme(1,1)
        sns.lineplot(x ='Sampl%', y ='NMAE' , data= self.thr_res, color='#d95f02',
                        label='Threshold-based Scheduling')
        sns.scatterplot(x ='Sampl%', y= 'NMAE' , data= self.thr_res,
                    color='#d95f02', marker = 'o', s=50)

        sns.lineplot(x ='Sampl%',y = 'NMAE' , data= self.sim_res, color='#1b9e77',
                    linestyle='--', label='Similarity-based Scheduling')
        sns.scatterplot(x ='Sampl%', y ='NMAE' , data= self.sim_res, color='#1b9e77',
                             marker = 'o', s=50)

        ax_.xaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Sampling Reduction%')

        plt.savefig(filename , bbox_inches='tight')
        plt.show()

    def plot_results3(self, filename):
        """
        Plot results, using the hashed plot bars.
        """
        _, ax_1 = plt.subplots(2,2,figsize=(15, 15))

        sns.barplot(x ='std', y ='RMSE' , data= self.thr_res, color='#d95f02',
                        label='Threshold-based', ax =ax_1[0,0])
        sns.barplot(x = 'std' , y='Sampl%',  data= self.thr_res, color='#d95f02',
                        alpha=0.7, hatch='xx',
                        label='%Reduction', ax=ax_1[1,0])
        sns.barplot(x='ThD', y='RMSE' , data= self.sim_res, color='#1b9e77',
                                label='Similarity-based', ax =ax_1[0,1])
        sns.barplot(x='ThD' ,y='Sampl%',  data= self.sim_res, color='#1b9e77', alpha=0.7,
                        hatch='xx',label='%Reduction', ax=ax_1[1,1])

        for ax_ in [ax_1[0,0], ax_1[0,1]]:
            ax_.xaxis.set_tick_params(which='both', rotation=90)
            ax_.set_ylim(0, )

        for ax_ in [ax_1[1,0], ax_1[1,1]]:
            ax_.set_ylim(0, 1)
            ax_.yaxis.set_major_formatter(PercentFormatter(1))
            ax_.xaxis.set_tick_params(which='both', rotation=90)
            ax_.set(ylabel='Sampling Reduction%')

        for ax_ in [ax_1[0,1] , ax_1[1,1]]:
            ax_.set(ylabel=None)
            ax_.set(yticklabels=[])

        for ax_ in [ax_1[0,0], ax_1[1,0]]:
            ax_.set(xlabel='Absolute Threshold')

        for ax_ in [ax_1[0,1], ax_1[1,1]]:
            ax_.set(xlabel='Similarity Threshold')


        plt.savefig(filename , bbox_inches='tight')
        plt.show()
