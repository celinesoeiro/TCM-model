from tcm_params import TCM_model_parameters

import math
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

dt = TCM_model_parameters()['dt']
fs = TCM_model_parameters()['sampling_frequency']
sim_steps = TCM_model_parameters()['simulation_steps']

lowcut = TCM_model_parameters()['beta_low']
highcut = TCM_model_parameters()['beta_high']
dbs_begin = TCM_model_parameters()['dbs_begin']
dbs_end = TCM_model_parameters()['dbs_end']

time_arr = np.arange(0, sim_steps + 1, fs, dtype=int)
xlabels = [f'{int(x/fs)}' for x in time_arr]

def plot_heat_map(matrix_normal, matrix_PD): 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,7))
    
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Conection matrix')
    
    sns.heatmap(matrix_normal, 
                vmin=-1, vmax=1, 
                yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
                annot=True, 
                fmt=".3f", 
                linewidth=.75,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                ax=ax1,
                )
    ax1.set(xlabel="", ylabel="")
    ax1.xaxis.tick_top()
    ax1.set_title('normal condition')
    
    sns.heatmap(matrix_PD, 
                vmin=-1, vmax=1, 
                yticklabels=['S', 'M', 'D', 'CI', 'TC', 'TR'], 
                annot=True, 
                fmt=".3f", 
                linewidth=.75,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                ax=ax2,
                )
    ax2.set(xlabel="", ylabel="")
    ax2.xaxis.tick_top()
    ax2.set_title('parkinsonian condition')
    
    plt.savefig('results/connection-matrix.png')
    
    # plt.show()
    
def plot_voltages(n_neurons, voltage, title, neuron_types):
    n_rows = int(n_neurons/2)
    
    fig, axs = plt.subplots(n_rows, 2, sharex=True, figsize=(n_neurons + 10,n_neurons + 10))
        
    fig.suptitle(title)    
    
    for i in range(n_neurons):
        column = 0
        row = math.floor(i/2)
                
        if (i%2 == 0):
            column = 0
        else:
            column = 1
        
        neuron_type = neuron_types[i]
        
        axs[row,column].set_title(f'neuron {i + 1} - {neuron_type}')
        axs[row,column].plot(voltage[i])
    
    plt.savefig(f'results/{title}.png')
    # plt.show()
    
def showPSD(signal, n):
    (f, S) = scipy.signal.welch(signal[n], fs)
    
    plt.semilogy(f, S)
    plt.ylim([1e-3, 1e2])
    plt.xlim([0, 50])
    plt.xticks([0,5,10,15,20,25,30,35,40,45,50])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title(f'neuron - {n}')
    plt.savefig('results/connection-matrix.png')
    # plt.show()
    
def plot_LFP(lfp, title):
    new_time= np.transpose(np.arange(len(lfp)))
    
    plt.figure(figsize=(15, 15))
    
    plt.title(title)

    plt.plot(new_time, lfp)
    
    # Set the x-axis label
    plt.xlabel('Time')
    plt.ylabel('LFP')
    
    plt.savefig(f'results/{title}.png')
    # Show the plot
    # plt.show()
    
def plot_LFPs(LFP_S, LFP_M, LFP_D, LFP_CI, LFP_TC, LFP_TR, title):
    new_time= np.transpose(np.arange(len(LFP_S)))
    
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(15, 10))
    
    ax1.plot(new_time, LFP_S)
    ax2.plot(new_time, LFP_M)
    ax3.plot(new_time, LFP_D)
    ax4.plot(new_time, LFP_CI)
    ax5.plot(new_time, LFP_TC)
    ax6.plot(new_time, LFP_TR)
    
    ax1.set_title('S')
    ax2.set_title('M')
    ax3.set_title('D')
    ax4.set_title('CI')
    ax5.set_title('TC')
    ax6.set_title('TR')
    
    fig.suptitle(title)
        
    plt.show()
    
def plot_I_DBS(I, title):    
    plt.figure()
    plt.title(f'{title}')
    plt.xticks(time_arr, labels=xlabels)
    plt.ylabel('current (mA)')
    plt.xlabel('time (s)')
    plt.plot(I)
    plt.savefig(f'results/{title}.png')
    # plt.show()
    
def plot_BP_filter(signal, dbs_freq):
    x_offset = dbs_begin/2
    max_value = signal.max()
    y_pos_begin = max_value - max_value/5
    y_pos_end = max_value - max_value/4

    plt.figure(figsize=(30, 10))
    plt.xticks(time_arr, labels=xlabels)
    plt.plot(signal)
    plt.annotate('begin DBS', xy=(dbs_begin, y_pos_begin), xytext=(dbs_begin + x_offset, y_pos_end),
                  arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'color':"black"}
                  ,horizontalalignment='center', fontsize=16)
    plt.annotate('end DBS', xy=(dbs_end, y_pos_begin), xytext=(dbs_end + x_offset, y_pos_end),
                  arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'color':"black"}
                  ,horizontalalignment='center', fontsize=16)
    plt.legend([f'Parkinsonian - DBS {dbs_freq}', 'Normal'], fontsize=16)
    plt.title(f'LFP bandpass filtered - ${lowcut} - ${highcut}', fontsize=16)
    plt.ylabel('potential (uV)')
    plt.xlabel('time (s)')
    plt.savefig(f'results/LFP_bandpass_filtered-{lowcut}-{highcut}.png')
    # plt.show()
    
def plot_BP_filter_normal(signal):
    plt.figure(figsize=(30, 10))
    plt.xticks(time_arr, labels=xlabels)
    plt.title(f'LFP bandpass filtered - ${lowcut} - ${highcut}', fontsize=16)
    plt.plot(signal)
    plt.savefig(f'results/LFP_bandpass_filtered-{lowcut}-{highcut}.png')
    # plt.show()
    
def plot_PSD_DBS(f, S, dbs_freq):
    x_arr = np.arange(0, 101, 10)
    
    plt.figure(figsize=(21, 10))
    plt.semilogy(f, S)
    plt.ylim([1e-3, 1e8])
    plt.xlim([0, 100])
    plt.xticks(x_arr)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title(f'PSD - {dbs_freq}')
    plt.savefig(f'results/PSD-{dbs_freq}.png')
    # plt.show()
    
def plot_PSD(f, S):
    x_arr = np.arange(0, 101, 10)
    
    plt.figure(figsize=(21, 10))
    plt.semilogy(f, S)
    plt.ylim([1e-3, 1e8])
    plt.xlim([0, 100])
    plt.xticks(x_arr)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('PSD')
    plt.savefig('results/PSD.png')
    # plt.show()
    
# =============================================================================
# RASTER
# =============================================================================
def layer_raster_plot(n, AP, sim_steps, layer_name, dt):
    fig, ax1 = plt.subplots()
    
    fig.canvas.manager.set_window_title(f'Raster plot - {layer_name}')

    for i in range(n):
        y_values = np.full_like(AP[i], i + 1)
        ax1.scatter(x=AP[i], y=y_values, color='black', s=1)
        
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                    alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Raster plot - {layer_name}',
        xlabel='time (s)',
        ylabel='neurons',
    )
    
    y_labels_vec = np.arange(0, n + 1, 1, dtype=int)
     
    ax1.set_ylim(1, n + 1)
    ax1.set_yticks(y_labels_vec)
    ax1.set_yticklabels(y_labels_vec)
    ax1.set_xlim(0, sim_steps)
    ax1.set_xticks(time_arr, labels=xlabels)
    plt.savefig(f'results/Raster_plot-{layer_name}.png')
    # plt.show()
    
def plot_raster(
    sim_steps,
    dt,
    chop_till, 
    n_TR, 
    n_TC, 
    n_CI, 
    n_D, 
    n_M, 
    n_S, 
    n_total,
    n_CI_LTS,
    n_D_IB,
    n_S_IB,
    spike_times_TR, 
    spike_times_TC, 
    spike_times_CI, 
    spike_times_D, 
    spike_times_M,
    spike_times_S):
    
    TR_lim = n_TR
    TC_lim = TR_lim + n_TC
    CI_lim = TC_lim + n_CI
    CI_FS_lim = CI_lim - n_CI_LTS
    D_lim = CI_lim + n_D
    D_RS_lim = D_lim - n_D_IB
    M_lim = D_lim + n_M
    S_lim = M_lim + n_S
    S_RS_lim = S_lim - n_S_IB
    
    spike_TR_clean = np.zeros((n_TR, sim_steps - chop_till))
    spike_TC_clean = np.zeros((n_TC, sim_steps - chop_till))
    spike_CI_clean = np.zeros((n_CI, sim_steps - chop_till))
    spike_D_clean = np.zeros((n_D, sim_steps - chop_till))
    spike_M_clean = np.zeros((n_M, sim_steps - chop_till))
    spike_S_clean = np.zeros((n_S, sim_steps - chop_till))
    
    for i in range(n_TR):
        spike_TR_clean[i] = spike_times_TR[i][chop_till:]
        
    for i in range(n_TC):
        spike_TC_clean[i] = spike_times_TC[i][chop_till:]
        spike_CI_clean[i] = spike_times_CI[i][chop_till:]
        spike_D_clean[i] = spike_times_D[i][chop_till:]
        spike_M_clean[i] = spike_times_M[i][chop_till:]
        spike_S_clean[i] = spike_times_S[i][chop_till:]
    
    spikes = np.concatenate([spike_TR_clean, spike_TC_clean, spike_CI_clean, spike_D_clean, spike_M_clean, spike_S_clean])
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Raster plot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        
    plt.title('Raster plot')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Raster plot',
        xlabel='time (s)',
        ylabel='neurons',
    )
        
    for i in range(n_total):  
        y_values = np.full_like(spikes[i], i + 1)
        ax1.scatter(x=spikes[i], y=y_values, color='black', s=0.5)
        
    ax1.set_ylim(1, n_total + 1)
    ax1.set_yticks([0, 
                   TR_lim, 
                   TC_lim, 
                   CI_lim, 
                   CI_FS_lim, 
                   CI_FS_lim, 
                   D_RS_lim, 
                   D_RS_lim, 
                   D_lim, 
                   M_lim, 
                   S_RS_lim, 
                   S_RS_lim, 
                   S_lim])
    ax1.set_yticklabels(['',
                        'TR',
                        'TC',
                        'CI - FS',
                        'CI - LTS',
                        'CI',
                        'D - RS',
                        'D - IB',
                        'D', 
                        'M - RS', 
                        'S - RS', 
                        'S - IB', 
                        'S',
                        ])
    
    # For dt = 0.1
    multiplier = 1000
    lim_down = chop_till
    lim_up = sim_steps + multiplier*dt
    # new_arr = np.arange(lim_down, lim_up, multiplier)
    
    # Transforming flot array to int array
    # x_ticks = list(map(int,new_arr/multiplier))
    
    ax1.set_xlim(lim_down, lim_up)
    ax1.set_xticks(time_arr, labels=xlabels)
    
    # TR neurons
    ax1.hlines(y = TR_lim, xmin=0, xmax=sim_steps, color = 'b', linestyle='solid' )
    # TC neurons
    ax1.hlines(y = TC_lim, xmin=0, xmax=sim_steps, color = 'g', linestyle='solid' )
    # CI neurons
    ax1.hlines(y = CI_lim, xmin=0, xmax=sim_steps, color = 'r', linestyle='solid' )
    ax1.hlines(y = CI_FS_lim, xmin=0, xmax=sim_steps, color = 'lightcoral', linestyle='solid')
    # D neurons
    ax1.hlines(y = D_lim, xmin=0, xmax=sim_steps, color = 'c', linestyle='solid' )
    ax1.hlines(y = D_RS_lim, xmin=0, xmax=sim_steps, color = 'paleturquoise', linestyle='solid' )
    # M neurons
    ax1.hlines(y = M_lim, xmin=0, xmax=sim_steps, color = 'm', linestyle='solid' )
    # S neurons
    ax1.hlines(y = S_lim, xmin=0, xmax=sim_steps, color = 'gold', linestyle='solid' )
    ax1.hlines(y = S_RS_lim, xmin=0, xmax=sim_steps, color = 'khaki', linestyle='solid' )
    plt.savefig('results/Raster_plot.png')
    # plt.show()
    