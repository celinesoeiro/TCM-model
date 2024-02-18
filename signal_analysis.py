"""
Created on Sun Feb 18 11:23:18 2024

@author: celinesoeiro
"""

from tcm_params import TCM_model_parameters, coupling_matrix_normal, coupling_matrix_PD
from model_plots import plot_heat_map, plot_raster, plot_BP_filter, plot_PSD, plot_BP_filter_normal
from model_functions import LFP, butter_bandpass_filter, PSD

from TR_nucleus import TR_nucleus as TR_nucleus_normal
from TC_nucleus import TC_nucleus as TC_nucleus_normal
from S_nucleus import S_nucleus as S_nucleus_normal
from M_nucleus import M_nucleus as M_nucleus_normal
from D_nucleus import D_nucleus as D_nucleus_normal
from CI_nucleus import CI_nucleus as CI_nucleus_normal

from TR_nucleus_PD import TR_nucleus as TR_nucleus_PD
from TC_nucleus_PD import TC_nucleus as TC_nucleus_PD
from S_nucleus_PD import S_nucleus as S_nucleus_PD
from M_nucleus_PD import M_nucleus as M_nucleus_PD
from D_nucleus_PD import D_nucleus as D_nucleus_PD
from CI_nucleus_PD import CI_nucleus as CI_nucleus_PD

from dbs_compare import dbs

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from random import seed, random

seed(1)
random_factor = random()

# =============================================================================
# INITIAL VALUES
# =============================================================================
print("-- Initializing the global values")
neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_per_structure = TCM_model_parameters()['neuron_per_structure']
neuron_params = TCM_model_parameters()['neuron_paramaters']
neuron_types_per_structure = TCM_model_parameters()['neuron_types_per_structure']

currents = TCM_model_parameters()['currents_per_structure']
p = TCM_model_parameters()['synapse_total_params']

noise = TCM_model_parameters()['noise']

# Neuron quantities
n_S = neuron_quantities['S']
n_M = neuron_quantities['M']
n_D = neuron_quantities['D']
n_CI = neuron_quantities['CI']
n_TR = neuron_quantities['TR']
n_TC = neuron_quantities['TC']
n_Hyper = neuron_quantities['HD']
n_total = neuron_quantities['total']

n_CI_FS = neuron_per_structure['neurons_ci_1']
n_CI_LTS = neuron_per_structure['neurons_ci_2']
n_D_RS = neuron_per_structure['neurons_d_1']
n_D_IB = neuron_per_structure['neurons_d_2']
n_S_RS = neuron_per_structure['neurons_s_1']
n_S_IB = neuron_per_structure['neurons_s_2']

vr = TCM_model_parameters()['vr']

dt = TCM_model_parameters()['dt']
sim_time = TCM_model_parameters()['simulation_time']
T = TCM_model_parameters()['simulation_time_ms']

sim_steps = TCM_model_parameters()['simulation_steps']
time_v = TCM_model_parameters()['time_vector']
time = np.arange(1, sim_steps)

lowcut = TCM_model_parameters()['beta_low']
highcut = TCM_model_parameters()['beta_high']
fs = TCM_model_parameters()['sampling_frequency']
dbs_begin = TCM_model_parameters()['dbs_begin']
dbs_end = TCM_model_parameters()['dbs_end']

# =============================================================================
# COUPLING MATRIXES
# =============================================================================
# Weight Matrix Normal Condition
Z_N = coupling_matrix_normal()['matrix']

# Weight Matrix Parkinsonian Desease Condition
Z_PD = coupling_matrix_PD()['matrix']

# normalizing Normal coupling matrix
Z_N_norm = Z_N/np.linalg.norm(Z_N)

# normalizing PD coupling matrix
Z_PD_norm = Z_PD/np.linalg.norm(Z_PD)

print("-- Printing the coupling matrixes")

CM_Normal = pd.DataFrame(Z_N_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])
CM_PD = pd.DataFrame(Z_PD_norm, columns=['S', 'M', 'D', 'CI', 'TC', 'TR'])

plot_heat_map(matrix_normal = CM_Normal, matrix_PD = CM_PD)

# Creating x labels
time_arr = np.arange(0, sim_steps + 1, fs, dtype=int)
xlabels = [f'{int(x/fs)}' for x in time_arr]

# =============================================================================
# NORMAL CONDITION
# =============================================================================
def normal_condition():
    print("-- Initializing the neuron variables")
    ## Post-Synaptic Currents
    PSC_S = np.zeros((1, sim_steps))
    PSC_M = np.zeros((1, sim_steps))
    PSC_D = np.zeros((1, sim_steps))
    PSC_CI = np.zeros((1, sim_steps))
    PSC_TC = np.zeros((1, sim_steps))
    PSC_TR = np.zeros((1, sim_steps))
    ## Thalamus-Cortex coupling params
    PSC_T_D = np.zeros((1, sim_steps)) # from Thalamus to D
    PSC_D_T = np.zeros((1, sim_steps)) # from D to Thalamus

    ## S
    AP_S = np.zeros((n_S, sim_steps))

    v_S = np.zeros((n_S, sim_steps))
    u_S = np.zeros((n_S, sim_steps)) 
    for i in range(n_S):    
        v_S[i][0] = vr
        u_S[i][0] = neuron_params['b_S'][0][0]*vr

    u_S_syn = np.zeros((1, p))
    R_S_syn = np.ones((1, p))
    I_S_syn = np.zeros((1, p))

    del i

    ## M
    AP_M = np.zeros((n_M, sim_steps))

    v_M = np.zeros((n_M, sim_steps))
    u_M = np.zeros((n_M, sim_steps)) 
    for i in range(n_M):    
        v_M[i][0] = vr
        u_M[i][0] = neuron_params['b_M'][0][0]*vr

    u_M_syn = np.zeros((1, p))
    R_M_syn = np.ones((1, p))
    I_M_syn = np.zeros((1, p))

    del i

    ## D
    AP_D = np.zeros((n_D, sim_steps))

    v_D = np.zeros((n_D, sim_steps))
    u_D = np.zeros((n_D, sim_steps)) 
    for i in range(n_D):    
        v_D[i][0] = vr
        u_D[i][0] = neuron_params['b_D'][0][0]*vr

    ## D - Self
    u_D_syn = np.zeros((1, p))
    R_D_syn = np.ones((1, p))
    I_D_syn = np.zeros((1, p))

    del i

    ## CI
    AP_CI = np.zeros((n_CI, sim_steps))

    v_CI = np.zeros((n_CI, sim_steps))
    u_CI = np.zeros((n_CI, sim_steps)) 
    for i in range(n_CI):    
        v_CI[i][0] = vr
        u_CI[i][0] = neuron_params['b_CI'][0][0]*vr
        
    u_CI_syn = np.zeros((1, p))
    R_CI_syn = np.ones((1, p))
    I_CI_syn = np.zeros((1, p))

    del i

    ## TC
    AP_TC = np.zeros((n_TC, sim_steps))

    v_TC = np.zeros((n_TC, sim_steps))
    u_TC = np.zeros((n_TC, sim_steps)) 
    for i in range(n_TC):    
        v_TC[i][0] = vr
        u_TC[i][0] = neuron_params['b_TC'][0][0]*vr
        
    u_TC_syn = np.zeros((1, p))
    R_TC_syn = np.ones((1, p))
    I_TC_syn = np.zeros((1, p))

    del i

    ## TR
    AP_TR = np.zeros((n_TR, sim_steps))

    v_TR = np.zeros((n_TR, sim_steps))
    u_TR = np.zeros((n_TR, sim_steps)) 
    for i in range(n_TR):    
        v_TR[i][0] = vr
        u_TR[i][0] = neuron_params['b_TR'][0][0]*vr
        
    u_TR_syn = np.zeros((1, p))
    R_TR_syn = np.ones((1, p))
    I_TR_syn = np.zeros((1, p))

    # =============================================================================
    # MAIN
    # =============================================================================
    print("-- Running the model")
    for t in time:
    # =============================================================================
    #     TR
    # =============================================================================
        v_TR, u_TR, PSC_TR, u_TR_syn, I_TR_syn, R_TR_syn = TR_nucleus_normal(t, v_TR, u_TR, AP_TR, PSC_TR, PSC_TC, PSC_CI, PSC_D_T, PSC_M, PSC_S, u_TR_syn, R_TR_syn, I_TR_syn)
    # =============================================================================
    #     TC
    # =============================================================================
        v_TC, u_TC, PSC_TC, u_TC_syn, I_TC_syn, R_TC_syn, PSC_T_D = TC_nucleus_normal(t, v_TC, u_TC, AP_TC, PSC_TC, PSC_S, PSC_M, PSC_D_T, PSC_TR, PSC_CI, PSC_T_D, R_TC_syn, u_TC_syn, I_TC_syn)
            
    # =============================================================================
    #     S
    # =============================================================================
        v_S, u_S, PSC_S, u_S_syn, I_S_syn, R_S_syn = S_nucleus_normal(t, v_S, u_S, AP_S, PSC_S, PSC_M, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_S_syn, R_S_syn, I_S_syn)
            
    # =============================================================================
    #     M
    # =============================================================================
        v_M, u_M, PSC_M, u_M_syn, I_M_syn, R_M_syn = M_nucleus_normal(t, v_M, u_M, AP_M, PSC_M, PSC_S, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_M_syn, R_M_syn, I_M_syn)
                
    # =============================================================================
    #     D
    # =============================================================================
        v_D, u_D, PSC_D, u_D_syn, I_D_syn, R_D_syn, PSC_D_T = D_nucleus_normal(t, v_D, u_D, AP_D, PSC_D, PSC_S, PSC_M, PSC_T_D, PSC_CI, PSC_TR, PSC_D_T, u_D_syn, R_D_syn, I_D_syn)
            
    # =============================================================================
    #     CI
    # =============================================================================
        v_CI, u_CI, PSC_CI, u_CI_syn, I_CI_syn, R_CI_syn = CI_nucleus_normal(t, v_CI, u_CI, AP_CI, PSC_CI, PSC_D, PSC_M, PSC_S, PSC_TC, PSC_TR, u_CI_syn, R_CI_syn, I_CI_syn)
        
    # =============================================================================
    # PLOTS
    # =============================================================================
    print("-- Plotting the results")

    # plot_voltages(n_neurons = n_S, voltage = v_S, title = "v - Layer S", neuron_types = neuron_types_per_structure['S'])
    # layer_raster_plot(n = n_S, AP = AP_S, sim_steps = sim_steps, layer_name = 'S', dt = dt)
    print('APs in S layer = ', np.count_nonzero(AP_S))

    # plot_voltages(n_neurons = n_M, voltage = v_M, title = "v - Layer M", neuron_types = neuron_types_per_structure['M'])
    # layer_raster_plot(n = n_M, AP = AP_M, sim_steps = sim_steps, layer_name = 'M', dt = dt)
    print('APs in M layer = ', np.count_nonzero(AP_M))

    # plot_voltages(n_neurons = n_D, voltage = v_D, title = "v - Layer D", neuron_types=neuron_types_per_structure['D'])
    # layer_raster_plot(n = n_D, AP = AP_D, sim_steps = sim_steps, layer_name = 'D', dt = dt)
    print('APs in D layer = ', np.count_nonzero(AP_D))

    # plot_voltages(n_neurons = n_CI, voltage = v_CI, title = "Layer CI", neuron_types=neuron_types_per_structure['CI'])
    # layer_raster_plot(n = n_CI, AP = AP_CI, sim_steps = sim_steps, layer_name = 'CI', dt = dt)
    print('APS in CI layer = ',np.count_nonzero(AP_CI))

    # plot_voltages(n_neurons = n_TC, voltage = v_TC, title = "TC", neuron_types=neuron_types_per_structure['TC'])
    # layer_raster_plot(n = n_TC, AP = AP_TC, sim_steps = sim_steps, layer_name = 'TC', dt = dt)
    print('APS in TC layer = ',np.count_nonzero(AP_TC))

    # plot_voltages(n_neurons = n_TR, voltage = v_TR, title = "TR", neuron_types=neuron_types_per_structure['TR'])
    # layer_raster_plot(n = n_TR, AP = AP_TR, sim_steps = sim_steps, layer_name = 'TR', dt = dt)
    print('APS in TR layer = ',np.count_nonzero(AP_TR))

    plot_raster(
                sim_steps=sim_steps,
                dt = dt,
                chop_till = 0, 
                n_TR = n_TR, 
                n_TC = n_TC, 
                n_CI = n_CI, 
                n_D = n_D, 
                n_M = n_M, 
                n_S = n_S, 
                n_total = n_total,
                n_CI_LTS = n_CI_LTS,
                n_D_IB = n_D_IB,
                n_S_IB = n_S_IB,
                spike_times_TR = AP_TR, 
                spike_times_TC = AP_TC, 
                spike_times_CI = AP_CI, 
                spike_times_D = AP_D, 
                spike_times_M = AP_M,
                spike_times_S = AP_S
                )
    
    # =============================================================================
    # Signal analysis
    # =============================================================================
    print("-- Signal analysis")
    
    ## Getting the Local Field Potential
    LFP_normal_full_signal = LFP(PSC_D[0], PSC_CI[0])
    
    ## Bandpass filtering the LFP to get the beta waves - full signal
    beta_waves_normal_full = butter_bandpass_filter(LFP_normal_full_signal, lowcut, highcut, fs)
    plot_BP_filter_normal(beta_waves_normal_full)
    
    # Power Spectral Density - full signal
    f_normal_full, S_normal_full = PSD(beta_waves_normal_full, fs)
    plot_PSD(f_normal_full, S_normal_full)
    
    ## Get the signal in the places where DBS stimulus will be applied
    LFP_normal = LFP_normal_full_signal[dbs_begin:dbs_end]
    beta_waves_normal = butter_bandpass_filter(LFP_normal, lowcut, highcut, fs)
    plot_BP_filter_normal(beta_waves_normal)
    
    # Power Spectral Density
    f_normal, S_normal = PSD(beta_waves_normal, fs)
    plot_PSD(f_normal, S_normal)
    
    return f_normal, S_normal

def PD_condition():
    print("-- Initializing the neuron variables")
    ## Post-Synaptic Currents
    PSC_S = np.zeros((1, sim_steps))
    PSC_M = np.zeros((1, sim_steps))
    PSC_D = np.zeros((1, sim_steps))
    PSC_CI = np.zeros((1, sim_steps))
    PSC_TC = np.zeros((1, sim_steps))
    PSC_TR = np.zeros((1, sim_steps))
    ## Thalamus-Cortex coupling params
    PSC_T_D = np.zeros((1, sim_steps)) # from Thalamus to D
    PSC_D_T = np.zeros((1, sim_steps)) # from D to Thalamus

    ## S
    AP_S = np.zeros((n_S, sim_steps))

    v_S = np.zeros((n_S, sim_steps))
    u_S = np.zeros((n_S, sim_steps)) 
    for i in range(n_S):    
        v_S[i][0] = vr
        u_S[i][0] = neuron_params['b_S'][0][0]*vr

    u_S_syn = np.zeros((1, p))
    R_S_syn = np.ones((1, p))
    I_S_syn = np.zeros((1, p))

    del i

    ## M
    AP_M = np.zeros((n_M, sim_steps))

    v_M = np.zeros((n_M, sim_steps))
    u_M = np.zeros((n_M, sim_steps)) 
    for i in range(n_M):    
        v_M[i][0] = vr
        u_M[i][0] = neuron_params['b_M'][0][0]*vr

    u_M_syn = np.zeros((1, p))
    R_M_syn = np.ones((1, p))
    I_M_syn = np.zeros((1, p))

    del i

    ## D
    AP_D = np.zeros((n_D, sim_steps))

    v_D = np.zeros((n_D, sim_steps))
    u_D = np.zeros((n_D, sim_steps)) 
    for i in range(n_D):    
        v_D[i][0] = vr
        u_D[i][0] = neuron_params['b_D'][0][0]*vr

    ## D - Self
    u_D_syn = np.zeros((1, p))
    R_D_syn = np.ones((1, p))
    I_D_syn = np.zeros((1, p))

    del i

    ## CI
    AP_CI = np.zeros((n_CI, sim_steps))

    v_CI = np.zeros((n_CI, sim_steps))
    u_CI = np.zeros((n_CI, sim_steps)) 
    for i in range(n_CI):    
        v_CI[i][0] = vr
        u_CI[i][0] = neuron_params['b_CI'][0][0]*vr
        
    u_CI_syn = np.zeros((1, p))
    R_CI_syn = np.ones((1, p))
    I_CI_syn = np.zeros((1, p))

    del i

    ## TC
    AP_TC = np.zeros((n_TC, sim_steps))

    v_TC = np.zeros((n_TC, sim_steps))
    u_TC = np.zeros((n_TC, sim_steps)) 
    for i in range(n_TC):    
        v_TC[i][0] = vr
        u_TC[i][0] = neuron_params['b_TC'][0][0]*vr
        
    u_TC_syn = np.zeros((1, p))
    R_TC_syn = np.ones((1, p))
    I_TC_syn = np.zeros((1, p))

    del i

    ## TR
    AP_TR = np.zeros((n_TR, sim_steps))

    v_TR = np.zeros((n_TR, sim_steps))
    u_TR = np.zeros((n_TR, sim_steps)) 
    for i in range(n_TR):    
        v_TR[i][0] = vr
        u_TR[i][0] = neuron_params['b_TR'][0][0]*vr
        
    u_TR_syn = np.zeros((1, p))
    R_TR_syn = np.ones((1, p))
    I_TR_syn = np.zeros((1, p))

    tr_aux = 0
    # =============================================================================
    # MAIN
    # =============================================================================
    print("-- Running model")
    for t in time:
    # =============================================================================
    #     TR
    # =============================================================================
        v_TR, u_TR, PSC_TR, u_TR_syn, I_TR_syn, R_TR_syn, tr_aux = TR_nucleus_PD(t, v_TR, u_TR, AP_TR, PSC_TR, PSC_TC, PSC_CI, PSC_D_T, PSC_M, PSC_S, u_TR_syn, R_TR_syn, I_TR_syn, tr_aux)
        
    # =============================================================================
    #     TC
    # =============================================================================
        v_TC, u_TC, PSC_TC, u_TC_syn, I_TC_syn, R_TC_syn, PSC_T_D = TC_nucleus_PD(t, v_TC, u_TC, AP_TC, PSC_TC, PSC_S, PSC_M, PSC_D_T, PSC_TR, PSC_CI, PSC_T_D, R_TC_syn, u_TC_syn, I_TC_syn)
            
    # =============================================================================
    #     S
    # =============================================================================
        v_S, u_S, PSC_S, u_S_syn, I_S_syn, R_S_syn = S_nucleus_PD(t, v_S, u_S, AP_S, PSC_S, PSC_M, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_S_syn, R_S_syn, I_S_syn)
            
    # =============================================================================
    #     M
    # =============================================================================
        v_M, u_M, PSC_M, u_M_syn, I_M_syn, R_M_syn = M_nucleus_PD(t, v_M, u_M, AP_M, PSC_M, PSC_S, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_M_syn, R_M_syn, I_M_syn)
                
    # =============================================================================
    #     D
    # =============================================================================
        v_D, u_D, PSC_D, u_D_syn, I_D_syn, R_D_syn, PSC_D_T = D_nucleus_PD(t, v_D, u_D, AP_D, PSC_D, PSC_S, PSC_M, PSC_T_D, PSC_CI, PSC_TR, PSC_D_T, u_D_syn, R_D_syn, I_D_syn)
            
    # =============================================================================
    #     CI
    # =============================================================================
        v_CI, u_CI, PSC_CI, u_CI_syn, I_CI_syn, R_CI_syn = CI_nucleus_PD(t, v_CI, u_CI, AP_CI, PSC_CI, PSC_D, PSC_M, PSC_S, PSC_TC, PSC_TR, u_CI_syn, R_CI_syn, I_CI_syn)
        
    # =============================================================================
    # PLOTS
    # =============================================================================
    print("-- Plotting results")

    # plot_voltages(n_neurons = n_S, voltage = v_S, title = "v - Layer S", neuron_types = neuron_types_per_structure['S'])
    # layer_raster_plot(n = n_S, AP = AP_S, sim_steps = sim_steps, layer_name = 'S', dt = dt)
    print('APs in S layer = ', np.count_nonzero(AP_S))

    # plot_voltages(n_neurons = n_M, voltage = v_M, title = "v - Layer M", neuron_types = neuron_types_per_structure['M'])
    # layer_raster_plot(n = n_M, AP = AP_M, sim_steps = sim_steps, layer_name = 'M', dt = dt)
    print('APs in M layer = ', np.count_nonzero(AP_M))

    # plot_voltages(n_neurons = n_D, voltage = v_D, title = "v - Layer D", neuron_types=neuron_types_per_structure['D'])
    # layer_raster_plot(n = n_D, AP = AP_D, sim_steps = sim_steps, layer_name = 'D', dt = dt)
    print('APs in D layer = ', np.count_nonzero(AP_D))

    # plot_voltages(n_neurons = n_CI, voltage = v_CI, title = "Layer CI", neuron_types=neuron_types_per_structure['CI'])
    # layer_raster_plot(n = n_CI, AP = AP_CI, sim_steps = sim_steps, layer_name = 'CI', dt = dt)
    print('APS in CI layer = ',np.count_nonzero(AP_CI))

    # plot_voltages(n_neurons = n_TC, voltage = v_TC, title = "TC", neuron_types=neuron_types_per_structure['TC'])
    # layer_raster_plot(n = n_TC, AP = AP_TC, sim_steps = sim_steps, layer_name = 'TC', dt = dt)
    print('APS in TC layer = ',np.count_nonzero(AP_TC))

    # plot_voltages(n_neurons = n_TR, voltage = v_TR, title = "TR", neuron_types=neuron_types_per_structure['TR'])
    # layer_raster_plot(n = n_TR, AP = AP_TR, sim_steps = sim_steps, layer_name = 'TR', dt = dt)
    print('APS in TR layer = ',np.count_nonzero(AP_TR))

    plot_raster(
                sim_steps=sim_steps,
                dt = dt,
                chop_till = 0, 
                n_TR = n_TR, 
                n_TC = n_TC, 
                n_CI = n_CI, 
                n_D = n_D, 
                n_M = n_M, 
                n_S = n_S, 
                n_total = n_total,
                n_CI_LTS = n_CI_LTS,
                n_D_IB = n_D_IB,
                n_S_IB = n_S_IB,
                spike_times_TR = AP_TR, 
                spike_times_TC = AP_TC, 
                spike_times_CI = AP_CI, 
                spike_times_D = AP_D, 
                spike_times_M = AP_M,
                spike_times_S = AP_S
                )

    print("-- Done!")

    # =============================================================================
    # Signal analysis
    # =============================================================================
    print("-- Signal analysis")

    fs = TCM_model_parameters()['sampling_frequency']

    ## Getting the Local Field Potential
    LFP_D_PD = LFP(PSC_D[0], PSC_CI[0])

    ## Bandpass filtering the LFP to get the beta waves
    beta_waves_PD_full = butter_bandpass_filter(LFP_D_PD, lowcut, highcut, fs)
    plot_BP_filter_normal(beta_waves_PD_full)

    # Power Spectral Density
    f_PD_full, S_PD_full = PSD(beta_waves_PD_full, fs)
    plot_PSD(f_PD_full, S_PD_full)

    PSD_signal_PD = LFP_D_PD[dbs_begin:dbs_end]
    beta_waves_PD = butter_bandpass_filter(PSD_signal_PD, lowcut, highcut, fs)
    # Power Spectral Density
    f_DBS_PD, S_DBS_PD = PSD(beta_waves_PD, fs)
    plot_PSD(f_DBS_PD, S_DBS_PD)
    
    return f_DBS_PD, S_DBS_PD

    print("-- Done!")

f_normal, s_normal = normal_condition()
f_PD, s_PD = PD_condition()
f_80, s_80 = dbs(80)
f_130, s_130 = dbs(130)
f_180, s_180 = dbs(180)

x_arr = np.arange(0, 81, 5)

plt.figure(figsize=(21, 10))
plt.semilogy(f_normal, s_normal, color="darkorange")
plt.semilogy(f_PD, s_PD, color="steelblue")
plt.semilogy(f_80, s_80, color="green")
plt.semilogy(f_130, s_130, color="red")
plt.semilogy(f_180, s_180, color="hotpink")
plt.legend([ 'normal condition','parkinsonian condition','DBS - 80Hz', 'DBS - 130Hz', 'DBS - 180Hz'], fontsize=22)
plt.ylim([1e-3, 1e8])
plt.xlim([0, 80])
plt.xticks(x_arr)
plt.xlabel('frequency (Hz)')
plt.ylabel('PSD [V**2/Hz]')
plt.title('PSD', fontsize=22)
plt.show()

