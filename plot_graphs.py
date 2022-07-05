import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from matplotlib.ticker import PercentFormatter

def set_plot_env():
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize=(10, 5))
    return fig, ax

def set_heatmap_env():
    sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    return fig, ax

def plot_node_Temperature(df, node_id):
    #fig, ax = set_plot_env()
    fig, axes  = plt.subplots(1,4,figsize=(20, 7))
    
    df = df[df['moteid'] == node_id]
    df = df.set_index('time_list')

    sns.scatterplot(x = "epoch", y = "Temperature", data = df, label='Original', marker='X', ax = axes[0])
    sns.scatterplot(x = "epoch", y="Temperature", data =df[::2], label ='50%', color='green', marker='X', ax=axes[1])
    sns.scatterplot(x = "epoch", y="Temperature", data =df[::4], label ='75%', color='red', marker='X', ax=axes[2])
    sns.scatterplot(x = "epoch", y="Temperature", data =df[::10], label ='90%', color='k', marker='X', ax=axes[3])

    plt.ylabel('Temperature')

    for ax in [axes[1], axes[2], axes[3]]:
        ax.set(ylabel=None)
        ax.set(yticklabels=[])  

    plt.savefig('../Figures/downsampling.pdf' , bbox_inches='tight')

def plot_temperature(df, file):
    """ 
    plot the temperature variable from all the nodes
    """
    fig, ax = set_plot_env()
    sns.lineplot('epoch' , 'Temperature', data=df, color = 'purple')
    plt.savefig(file , bbox_inches='tight')

def heatmap_hour_temperature(X):
    #Show the heat map for that  hour 
    colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']
    hour_start = np.arange(25804,28683 + 120 ,120)
    for i in range(len(hour_start) -1):
        fig, ax = set_heatmap_env()
        hour_v = np.arange(hour_start[i],hour_start[i+1])
        X_hour = X[hour_v]
        sns.heatmap(X_hour, yticklabels=True, vmin = 18, vmax = 38, cmap=sns.color_palette(colors))
        plt.savefig('./Figures/' + 'X_Hour' +str(i), bbox_inches='tight')

def plot_temperature_array(X):
    """
    plot the temperature for each mote in the array X
    """
    for mote in X.index:
        X_mote = X.loc[mote]
        fig, ax = set_plot_env()
        sns.scatterplot(X_mote.index, X_mote)
        plt.savefig('./Figures/X_mote/' + 'X_mote' + str(mote) + '.png', bbox_inches='tight')

def plot_density(df, file):
    fig, ax = set_heatmap_env()
    df.plot()
    plt.savefig(file, bbox_inches='tight')

def show_result(y_test, predicted):
    fig, ax = set_plot_env()
    plt.plot(y_test.index, predicted, 'o-', label="predicted")
    plt.plot(y_test.index, y_test, '.-', label="actual")
    plt.ylabel("Temperature")
    plt.legend()

def show_results_with_CI(X_test, prediction_mean, y_test, upper, lower, filename):
    fig, ax = set_plot_env()
    plt.plot(X_test, prediction_mean, color='red', label="predicted mean")
    plt.plot(X_test, y_test, '.-', label="actual")
    plt.fill_between(X_test, upper, lower, color='gray', label="95% CI")
    plt.ylabel("Temperature")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')

class plot_results():
    def __init__(self, threshold_results, similarity_results, VOI_results):
        self.thr_res = threshold_results
        self.sim_res = similarity_results
        self.voi_res = VOI_results

    def theme(self, n, m):
        sns.set_theme(style="white")
        sns.set_context("paper", font_scale=1.5)
        #sns.set(font="Times")
        fig, ax = plt.subplots(n,m,figsize=(12, 6))
        return fig, ax

    def plot_results1(self, filename):
        fig, ax = self.theme(1,1)

        sns.lineplot('Sampled', 'RMSE' , data= self.thr_res, color='#d95f02', label='Threshold-based Scheduling')
        sns.scatterplot('Sampled', 'RMSE' , data= self.thr_res,color='#d95f02', marker = 'o', s=50)

        sns.lineplot('Sampled', 'RMSE' , data= self.sim_res, color='#1b9e77', linestyle='--', label='Similarity-based Scheduling')
        sns.scatterplot('Sampled', 'RMSE' , data= self.sim_res, color='#1b9e77', marker = 'o', s=50)

        sns.lineplot('Sampled', 'RMSE' , data= self.voi_res, color='#7570b3', linestyle='--', label='VoI-based Scheduling')
        sns.scatterplot('Sampled', 'RMSE' , data= self.voi_res, color='#7570b3', marker = 'o', s=50)

        plt.savefig(filename, bbox_inches='tight')

    def plot_results2(self, filename):
        fig, ax = self.theme(1,1)
        sns.lineplot('Sampl%', 'NMAE' , data= self.thr_res, color='#d95f02', label='Threshold-based Scheduling')
        sns.scatterplot('Sampl%', 'NMAE' , data= self.thr_res, color='#d95f02', marker = 'o', s=50)

        sns.lineplot('Sampl%', 'NMAE' , data= self.sim_res, color='#1b9e77', linestyle='--', label='Similarity-based Scheduling')
        sns.scatterplot('Sampl%', 'NMAE' , data= self.sim_res, color='#1b9e77', marker = 'o', s=50)

        sns.lineplot('Sampl%', 'NMAE' , data= self.voi_res, color='#7570b3', linestyle='--', label='VoI-based Scheduling')
        sns.scatterplot('Sampl%', 'NMAE' , data= self.voi_res, color='#7570b3', marker = 'o', s=50)

        ax.xaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Sampling Reduction%')
        
        plt.savefig(filename , bbox_inches='tight')

    def plot_results3(self, filename):

        fig, ax1 = plt.subplots(2,3,figsize=(15, 15))
        #fig, ax1 =  self.theme(1,3)

        #ax21 = ax1[0].twinx()
        #ax22 = ax1[1].twinx()
        #ax23 = ax1[2].twinx()

        sns.barplot('std', 'NMAE' , data= self.thr_res, color='#d95f02',  label='Threshold-based', ax =ax1[0,0])
        sns.barplot('std' , 'Sampl%',  data= self.thr_res, color='#d95f02', alpha=0.7, hatch='xx', label='%Reduction', ax=ax1[1,0])
        sns.barplot('ThD', 'NMAE' , data= self.sim_res, color='#1b9e77',  label='Similarity-based', ax =ax1[0,1])
        sns.barplot('ThD' , 'Sampl%',  data= self.sim_res, color='#1b9e77', alpha=0.7, hatch='xx',label='%Reduction', ax=ax1[1,1])
        sns.barplot('ThD' , 'NMAE',  data= self.voi_res, color='#7570b3',label='VoI-based', ax=ax1[0,2])
        sns.barplot('ThD' , 'Sampl%',  data= self.voi_res, color='#7570b3', alpha=0.7, hatch='xx', label='%Reduction', ax=ax1[1,2])

        #width_scale = 0.45
        #for container in [ax21.containers[0], ax22.containers[0], ax23.containers[0]]:
         #   for bar in container:
          #      x = bar.get_x()
           #     w = bar.get_width()
            #    bar.set_x(x + w * (1- width_scale))
             #   bar.set_width(w * width_scale)

        for ax in [ax1[0,0], ax1[0,1], ax1[0,2]]:
            ax.xaxis.set_tick_params(which='both', rotation=90)
            ax.set_ylim(0, )

        for ax in [ax1[1,0], ax1[1,1], ax1[1,2]]:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_tick_params(which='both', rotation=90)
            ax.set(ylabel='Sampling Reduction%')

        for ax in [ax1[0,1], ax1[0,2] , ax1[1,1], ax1[1,2]]:
            ax.set(ylabel=None)
            ax.set(yticklabels=[])  

        for ax in [ax1[0,0], ax1[1,0]]:
            ax.set(xlabel='Absolute Threshold')

        for ax in [ax1[0,1], ax1[1,1]]:
            ax.set(xlabel='Similarity Threshold')
        
        for ax in [ax1[0,2], ax1[1,2]]:
            ax.set(xlabel='KL-divergence Threshold')

        #ax[0,1].set_title('Fig (a)')
        plt.savefig(filename , bbox_inches='tight')