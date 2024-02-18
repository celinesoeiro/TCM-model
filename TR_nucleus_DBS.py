"""
Created on Tue Feb  6 23:07:51 2024

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
syn_params = TCM_model_parameters()['synapse_params_inhibitory']

n_TR = neuron_quantities['TR']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

a_TR = neuron_params['a_TR']
b_TR = neuron_params['b_TR']
c_TR = neuron_params['c_TR']
d_TR = neuron_params['d_TR']

W_PD = coupling_matrix_PD()['weights']

W_TR_self = W_PD['W_II_tr']
W_TR_S = W_PD['W_IE_tr_s']
W_TR_M = W_PD['W_IE_tr_m']
W_TR_D = W_PD['W_IE_tr_d']
W_TR_TC = W_PD['W_IE_tr_tc']
W_TR_CI = W_PD['W_II_tr_ci']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

t_f_I = syn_params['t_f']
t_d_I = syn_params['t_d']
t_s_I = syn_params['t_s']
U_I = syn_params['U']
A_I = syn_params['distribution']

I_TR = currents['TR']

noise = TCM_model_parameters()['noise']

kisi_TR = noise['kisi_TR']
zeta_TR = noise['zeta_TR']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_TR = I_ps['TR']

syn_fid = TCM_model_parameters()['synaptic_fidelity_layers']
TR_fid = syn_fid['TR']

affected_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
TR_affected = affected_neurons['TR']

def TR_nucleus(t, v_TR, u_TR, AP_TR, PSC_TR, PSC_TC, PSC_CI, PSC_D_T, PSC_M, PSC_S, u_TR_syn, R_TR_syn, I_TR_syn, tr_aux, I_dbs):
    
    I_syn = np.zeros((1, n_TR))
    
    for tr in range(n_TR):
        if (tr >= 0 and tr <= TR_affected):
            dbs_I = 1*I_dbs[t - 1]
        else:
            dbs_I = 0
            
        tr_aux = tr
        v_TR_aux = 1*v_TR[tr][t - 1]
        u_TR_aux = 1*u_TR[tr][t - 1]
        AP_TR_aux = 0
                
        if (v_TR_aux >= vp + zeta_TR[tr][t - 1]):
            AP_TR_aux = 1
            AP_TR[tr][t] = t - 1
            v_TR_aux = v_TR[tr][t]
            v_TR[tr][t] = c_TR[0][tr]
            u_TR[tr][t] = u_TR_aux + d_TR[0][tr]
        else:
            AP_TR[tr][t] = 0
            AP_TR_aux = 0
            
            # Self feedback - Inhibitory
            coupling_TR_TR = W_TR_self[tr][0]*1*PSC_TR[0][t - td_wl - td_syn - 1]
            # Coupling TR to S - Excitatory 
            coupling_TR_S = W_TR_S[tr][0]*1*PSC_S[0][t - td_ct - td_syn - 1]
            # Coupling TR to M - Excitatory 
            coupling_TR_M = W_TR_M[tr][0]*1*PSC_M[0][t - td_ct - td_syn - 1]
            # Coupling TR to D - Excitatory 
            coupling_TR_D = W_TR_D[tr][0]*1*PSC_D_T[0][t - td_ct - td_syn - 1]
            # Coupling TR to CI - Inhibitory 
            coupling_TR_CI = W_TR_CI[tr][0]*1*PSC_CI[0][t - td_ct - td_syn - 1]
            # Coupling TR to TC - Inhibitory 
            coupling_TR_TC = W_TR_TC[tr][0]*1*PSC_TC[0][t - td_bl - td_syn - 1]
            
            dv_TR = izhikevich_dvdt(v = v_TR_aux, u = u_TR_aux, I = I_TR[tr])
            du_TR = izhikevich_dudt(v = v_TR_aux, u = u_TR_aux, a = a_TR[0][tr], b = b_TR[0][tr])
            
            coupling_cortex = (coupling_TR_S + coupling_TR_M + coupling_TR_D + coupling_TR_CI)/n_TR
            coupling_thalamus = (coupling_TR_TC + coupling_TR_TR)/n_TR
            bg_activity = kisi_TR[tr][t - 1] + I_ps_TR[0][t - td_wl - td_syn - 1] - I_ps_TR[1][t - td_wl - td_syn - 1]
        
            v_TR[tr][t] = v_TR_aux + dt*(dv_TR + coupling_cortex + coupling_thalamus + bg_activity + TR_fid*dbs_I)
            u_TR[tr][t] = u_TR_aux + dt*du_TR
            
        # Synapse - Within layer  
        syn_TR = tm_synapse_eq(u = u_TR_syn, R = R_TR_syn, I = I_TR_syn, AP = AP_TR_aux, t_f = t_f_I, t_d = t_d_I, t_s = t_s_I, U = U_I, A = A_I, dt = dt, p = p)
        
        R_TR_syn = 1*syn_TR['R']
        u_TR_syn = 1*syn_TR['u']
        I_TR_syn = 1*syn_TR['I']
        I_syn[0][tr] = 1*syn_TR['Ipost']
            
    PSC_TR[0][t] = np.sum(I_syn)
    return v_TR, u_TR, PSC_TR, u_TR_syn, I_TR_syn, R_TR_syn, tr_aux
