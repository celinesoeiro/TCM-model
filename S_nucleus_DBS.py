"""
Created on Mon Feb  5 22:18:42 2024

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

n_S = neuron_quantities['S']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

W_PS = coupling_matrix_PD()['weights']

W_S_self = W_PS['W_EE_s']
W_S_M = W_PS['W_EE_s_m']
W_S_D = W_PS['W_EE_s_d']
W_S_CI = W_PS['W_EI_s_ci']
W_S_TR = W_PS['W_EI_s_tr']
W_S_TC = W_PS['W_EE_s_tc']

a_S = neuron_params['a_S']
b_S = neuron_params['b_S']
c_S = neuron_params['c_S']
d_S = neuron_params['d_S']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_tc = TCM_model_parameters()['time_delay_thalamus_cortex']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

syn_fid = TCM_model_parameters()['synaptic_fidelity_layers']
S_fid = syn_fid['S']

affected_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
S_affected = affected_neurons['S']

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']

I_S = currents['S']

noise = TCM_model_parameters()['noise']

kisi_S = noise['kisi_S']
zeta_S = noise['zeta_S']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_S = I_ps['S']

def S_nucleus(t, v_S, u_S, AP_S, PSC_S, PSC_M, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_S_syn, R_S_syn, I_S_syn, I_dbs):
    
    I_syn = np.zeros((1, n_S))
    
    for s in range(n_S):
        if (s >= 0 and s <= S_affected):
            dbs_I = 1*I_dbs[t - 1]
        else:
            dbs_I = 0
            
        v_S_aux = 1*v_S[s][t - 1]
        u_S_aux = 1*u_S[s][t - 1]
        AP_S_aux = 0
                
        if (v_S_aux >= vp + zeta_S[s][t - 1]):
            AP_S_aux = 1
            AP_S[s][t] = t - 1
            v_S_aux = v_S[s][t]
            v_S[s][t] = c_S[0][s]
            u_S[s][t] = u_S_aux + d_S[0][s]
        else:
            AP_S[s][t] = 0
            AP_S_aux = 0
            
            # Self feedback - Inhibitory
            coupling_S_S = W_S_self[s][0]*1*PSC_S[0][t - td_wl - td_syn - 1]
            # Coupling S to M - Excitatory 
            coupling_S_M = W_S_M[s][0]*1*PSC_M[0][t - td_bl - td_syn - 1]
            # Coupling S to D - Excitatory 
            coupling_S_D = W_S_D[s][0]*1*PSC_D[0][t - td_bl - td_syn - 1]
            # Coupling S to CI - Inhibitory 
            coupling_S_CI = W_S_CI[s][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling S to TC - Excitatory
            coupling_S_TC = W_S_TC[s][0]*1*PSC_TC[0][t - td_tc - td_syn - 1]
            # Coupling S to TR - Excitatory
            coupling_S_TR = W_S_TR[s][0]*1*PSC_TR[0][t - td_tc - td_syn - 1]
            
            dv_S = izhikevich_dvdt(v = v_S_aux, u = u_S_aux, I = I_S[s])
            du_S = izhikevich_dudt(v = v_S_aux, u = u_S_aux, a = a_S[0][s], b = b_S[0][s])
            
            coupling_cortex = (coupling_S_S + coupling_S_M + coupling_S_D + coupling_S_CI)/n_S
            coupling_thalamus = (coupling_S_TC + coupling_S_TR)/n_S
            bg_activity = kisi_S[s][t - 1] + I_ps_S[0][t - td_wl - td_syn - 1] - I_ps_S[1][t - td_wl - td_syn - 1]
        
            v_S[s][t] = v_S_aux + dt*(dv_S + coupling_cortex + coupling_thalamus + bg_activity + S_fid*dbs_I)
            u_S[s][t] = u_S_aux + dt*du_S
            
        # Synapse - Within cortex  
        syn_S = tm_synapse_eq(u = u_S_syn, R = R_S_syn, I = I_S_syn, AP = AP_S_aux, t_f = t_f_E, t_d = t_d_E, t_s = t_s_E, U = U_E, A = A_E, dt = dt, p = p)
        
        R_S_syn = 1*syn_S['R']
        u_S_syn = 1*syn_S['u']
        I_S_syn = 1*syn_S['I']
        I_syn[0][s] = 1*syn_S['Ipost']
        
    PSC_S[0][t] = np.sum(I_syn)
    
    return v_S, u_S, PSC_S, u_S_syn, I_S_syn, R_S_syn

