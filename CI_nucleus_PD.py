"""
Created on Sat Feb  3 19:17:04 2024

@author: celinesoeiro
"""

import numpy as np

from tcm_params import TCM_model_parameters, coupling_matrix_PD
from model_functions import izhikevich_dudt, izhikevich_dvdt, tm_synapse_eq

neuron_quantities = TCM_model_parameters()['neuron_quantities']
neuron_per_structure = TCM_model_parameters()['neuron_per_structure']
neuron_params = TCM_model_parameters()['neuron_paramaters']
currents = TCM_model_parameters()['currents_per_structure']
dt = TCM_model_parameters()['dt']
syn_params = TCM_model_parameters()['synapse_params_excitatory']

n_CI = neuron_quantities['CI']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_tc = TCM_model_parameters()['time_delay_thalamus_cortex']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

W_PD = coupling_matrix_PD()['weights']

W_CI_self = W_PD['W_II_ci']
W_CI_S = W_PD['W_IE_ci_s']
W_CI_M = W_PD['W_IE_ci_m']
W_CI_D = W_PD['W_IE_ci_d']
W_CI_TR = W_PD['W_II_ci_tr']
W_CI_TC = W_PD['W_IE_ci_tc']

a_CI = neuron_params['a_CI']
b_CI = neuron_params['b_CI']
c_CI = neuron_params['c_CI']
d_CI = neuron_params['d_CI']

I_CI = currents['CI']

t_f_I = syn_params['t_f']
t_d_I = syn_params['t_d']
t_s_I = syn_params['t_s']
U_I = syn_params['U']
A_I = syn_params['distribution']

noise = TCM_model_parameters()['noise']

kisi_CI = noise['kisi_CI']
zeta_CI = noise['zeta_CI']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_CI = I_ps['CI']

def CI_nucleus(t, v_CI, u_CI, AP_CI, PSC_CI, PSC_D, PSC_M, PSC_S, PSC_TC, PSC_TR, u_CI_syn, R_CI_syn, I_CI_syn):
    
    I_syn = np.zeros((1, n_CI))
    
    for ci in range(n_CI):
        v_CI_aux = 1*v_CI[ci][t - 1]
        u_CI_aux = 1*u_CI[ci][t - 1]
        AP_CI_aux = 0
                
        if (v_CI_aux >= vp + zeta_CI[ci][t - 1]):
            AP_CI[ci][t] = t - 1
            AP_CI_aux = 1
            v_CI_aux = v_CI[ci][t]
            v_CI[ci][t] = c_CI[0][ci]
            u_CI[ci][t] = u_CI_aux + d_CI[0][ci]
        else:
            AP_CI_aux = 0
            AP_CI[ci][t] = 0
            
            # Self feeback - Inhibitory
            coupling_CI_CI = W_CI_self[ci][0]*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling CI to S - Inhibitory
            coupling_CI_S = W_CI_S[ci][0]*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling CI to M - Inhibitory
            coupling_CI_M = W_CI_M[ci][0]*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling CI to D - Inhibitory
            coupling_CI_D = W_CI_D[ci][0]*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling CI to TC - Inhibitory
            coupling_CI_TC = W_CI_TC[ci][0]*PSC_TC[0][t - td_tc - td_syn - 1]
            # Coupling CI to TR - Inhibitory
            coupling_CI_TR = W_CI_TR[ci][0]*PSC_TR[0][t - td_tc - td_syn - 1]
            
            dv_CI = izhikevich_dvdt(v = v_CI_aux, u = u_CI_aux, I = I_CI[ci])
            du_CI = izhikevich_dudt(v = v_CI_aux, u = u_CI_aux, a = a_CI[0][ci], b = b_CI[0][ci])
            
            coupling_cortex = (coupling_CI_M + coupling_CI_S + coupling_CI_CI + coupling_CI_D)/n_CI
            coupling_thalamus = (coupling_CI_TC + coupling_CI_TR)/n_CI
            bg_activity = kisi_CI[ci][t - 1] +  I_ps_CI[0][t - td_wl - td_syn - 1] - I_ps_CI[0][t - td_wl - td_syn - 1]
        
            v_CI[ci][t] = v_CI_aux + dt*(dv_CI + coupling_cortex + coupling_thalamus+ bg_activity)
            u_CI[ci][t] = u_CI_aux + dt*du_CI
            
        # Synapse        
        syn_CI = tm_synapse_eq(u = u_CI_syn, R = R_CI_syn, I = I_CI_syn, AP = AP_CI_aux, t_f = t_f_I, t_d = t_d_I, t_s = t_s_I, U = U_I, A = A_I,dt = dt, p = p)
        
        R_CI_syn = 1*syn_CI['R']
        u_CI_syn = 1*syn_CI['u']
        I_CI_syn = 1*syn_CI['I']
        I_syn[0][ci] = 1*syn_CI['Ipost']
        
    PSC_CI[0][t] = np.sum(I_syn)
    return v_CI, u_CI, PSC_CI, u_CI_syn, I_CI_syn, R_CI_syn


