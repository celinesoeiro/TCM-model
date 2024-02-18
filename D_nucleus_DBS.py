"""
Created on Tue Feb  6 22:16:46 2024

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

n_D = neuron_quantities['D']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

a_D = neuron_params['a_D']
b_D = neuron_params['b_D']
c_D = neuron_params['c_D']
d_D = neuron_params['d_D']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_tc = TCM_model_parameters()['time_delay_thalamus_cortex']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

W_N = coupling_matrix_PD()['weights']

W_D_self = W_N['W_EE_d']
W_D_S = W_N['W_EE_d_s']
W_D_M = W_N['W_EE_d_m']
W_D_CI = W_N['W_EI_d_ci']
W_D_TR = W_N['W_EI_d_tr']
W_D_TC = W_N['W_EE_d_tc']

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']
A_E_D_T = syn_params['distribution_D_T']

I_D = currents['D']

noise = TCM_model_parameters()['noise']

kisi_D = noise['kisi_D']
zeta_D = noise['zeta_D']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_D = I_ps['D']

syn_fid = TCM_model_parameters()['synaptic_fidelity_layers']
D_fid = syn_fid['D']

affected_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
D_affected = affected_neurons['D']

def D_nucleus(t, v_D, u_D, AP_D, PSC_D, PSC_S, PSC_M, PSC_T_D, PSC_CI, PSC_TR, PSC_D_T, u_D_syn, R_D_syn, I_D_syn, I_dbs):
    
    I_syn = np.zeros((1, n_D))
    I_syn_t = np.zeros((1, n_D))
    
    for d in range(n_D):
        if (d >= 0 and d <= D_affected):
            dbs_I = 1*I_dbs[0][t - 1]
        else:
            dbs_I = 1*I_dbs[1][t - 1]
            
        v_D_aux = 1*v_D[d][t - 1]
        u_D_aux = 1*u_D[d][t - 1]
        AP_D_aux = 0
                
        if (v_D_aux >= vp + zeta_D[d][t - 1]):
            AP_D_aux = 1
            AP_D[d][t] = t - 1
            v_D_aux = v_D[d][t]
            v_D[d][t] = c_D[0][d]
            u_D[d][t] = u_D_aux + d_D[0][d]
        else:
            AP_D[d][t] = 0
            AP_D_aux = 0
            
            # Self feedback - Inhibitory
            coupling_D_D = W_D_self[d][0]*1*PSC_D[0][t - td_wl - td_syn - 1]
            # Coupling D to S - Excitatory 
            coupling_D_S = W_D_S[d][0]*1*PSC_S[0][t - td_bl - td_syn - 1]
            # Coupling D to M - Excitatory 
            coupling_D_M = W_D_M[d][0]*1*PSC_M[0][t - td_bl - td_syn - 1]
            # Coupling D to CI - Inhibitory 
            coupling_D_CI = W_D_CI[d][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling D to TC - Excitatory
            coupling_D_TC = W_D_TC[d][0]*1*PSC_T_D[0][t - td_tc - td_syn - 1]
            # Coupling D to TR - Excitatory
            coupling_D_TR = W_D_TR[d][0]*1*PSC_TR[0][t - td_tc - td_syn - 1]
            
            dv_D = izhikevich_dvdt(v = v_D_aux, u = u_D_aux, I = I_D[d])
            du_D = izhikevich_dudt(v = v_D_aux, u = u_D_aux, a = a_D[0][d], b = b_D[0][d])
            
            coupling_cortex = (coupling_D_S + coupling_D_M + coupling_D_D + coupling_D_CI)/n_D
            coupling_thalamus = (coupling_D_TC + coupling_D_TR)/n_D
            bg_activity = kisi_D[d][t - 1] + I_ps_D[0][t - td_wl - td_syn - 1] - I_ps_D[1][t - td_wl - td_syn - 1]
        
            v_D[d][t] = v_D_aux + dt*(dv_D + coupling_cortex + coupling_thalamus + bg_activity + D_fid*dbs_I)
            u_D[d][t] = u_D_aux + dt*du_D
            
        # Synapse - Within cortex  
        syn_D = tm_synapse_eq(u = u_D_syn, R = R_D_syn, I = I_D_syn, AP = AP_D_aux, t_f = t_f_E, t_d = t_d_E, t_s = t_s_E, U = U_E, A = A_E, dt = dt, p = p)
        
        # Synapse - With Thalamus  
        syn_D_T = tm_synapse_eq(u = u_D_syn, R = R_D_syn, I = I_D_syn, AP = AP_D_aux, t_f = t_f_E, t_d = t_d_E, t_s = t_s_E, U = U_E, A = A_E_D_T, dt = dt, p = p)
        
        R_D_syn = 1*syn_D['R']
        u_D_syn = 1*syn_D['u']
        I_D_syn = 1*syn_D['I']
        I_syn[0][d] = 1*syn_D['Ipost']
        
        I_syn_t[0][d] = 1*syn_D_T['Ipost']
    
    PSC_D[0][t] = np.sum(I_syn)
    PSC_D_T[0][t] = np.sum(I_syn_t)
    
    return v_D, u_D, PSC_D, u_D_syn, I_D_syn, R_D_syn, PSC_D_T