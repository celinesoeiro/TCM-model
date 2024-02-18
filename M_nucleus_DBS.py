"""
Created on Tue Feb  6 21:56:03 2024

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

n_M = neuron_quantities['M']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

a_M = neuron_params['a_M']
b_M = neuron_params['b_M']
c_M = neuron_params['c_M']
d_M = neuron_params['d_M']

W_PD = coupling_matrix_PD()['weights']

W_M_self = W_PD['W_EE_m']
W_M_S = W_PD['W_EE_m_s']
W_M_D = W_PD['W_EE_m_d']
W_M_CI = W_PD['W_EI_m_ci']
W_M_TR = W_PD['W_EI_m_tr']
W_M_TC = W_PD['W_EE_m_tc']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_tc = TCM_model_parameters()['time_delay_thalamus_cortex']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']

I_M = currents['M']

noise = TCM_model_parameters()['noise']

kisi_M = noise['kisi_M']
zeta_M = noise['zeta_M']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_M = I_ps['M']

syn_fid = TCM_model_parameters()['synaptic_fidelity_layers']
M_fid = syn_fid['M']

affected_neurons = TCM_model_parameters()['neurons_connected_with_hyperdirect_neurons']
M_affected = affected_neurons['M']

def M_nucleus(t, v_M, u_M, AP_M, PSC_M, PSC_S, PSC_D, PSC_CI, PSC_TC, PSC_TR, u_M_syn, R_M_syn, I_M_syn, I_dbs):
    
    I_syn = np.zeros((1, n_M))
    
    for m in range(n_M):
        if (m >= 0 and m <= M_affected):
            dbs_I = 1*I_dbs[t - 1]
        else:
            dbs_I = 0
            
        v_M_aux = 1*v_M[m][t - 1]
        u_M_aux = 1*u_M[m][t - 1]
        AP_M_aux = 0
                
        if (v_M_aux >= vp + zeta_M[m][t - 1]):
            AP_M_aux = 1
            AP_M[m][t] = t - 1
            v_M_aux = v_M[m][t]
            v_M[m][t] = c_M[0][m]
            u_M[m][t] = u_M_aux + d_M[0][m]
        else:
            AP_M[m][t] = 0
            AP_M_aux = 0
            
            # Self feedback - Inhibitory
            coupling_M_M = W_M_self[m][0]*1*PSC_M[0][t - td_wl - td_syn - 1]
            # Coupling M to S - Excitatory 
            coupling_M_S = W_M_S[m][0]*1*PSC_S[0][t - td_bl - td_syn - 1]
            # Coupling M to D - Excitatory 
            coupling_M_D = W_M_D[m][0]*1*PSC_D[0][t - td_bl - td_syn - 1]
            # Coupling M to CI - Inhibitory 
            coupling_M_CI = W_M_CI[m][0]*1*PSC_CI[0][t - td_wl - td_syn - 1]
            # Coupling M to TC - Excitatory
            coupling_M_TC = W_M_TC[m][0]*1*PSC_TC[0][t - td_tc - td_syn - 1]
            # Coupling M to TR - Excitatory
            coupling_M_TR = W_M_TR[m][0]*1*PSC_TR[0][t - td_tc - td_syn - 1]
            
            dv_M = izhikevich_dvdt(v = v_M_aux, u = u_M_aux, I = I_M[m])
            du_M = izhikevich_dudt(v = v_M_aux, u = u_M_aux, a = a_M[0][m], b = b_M[0][m])
            
            coupling_cortex = (coupling_M_M + coupling_M_S + coupling_M_D + coupling_M_CI)/n_M
            coupling_thalamus = (coupling_M_TC + coupling_M_TR)/n_M
            bg_activity = kisi_M[m][t - 1] + I_ps_M[0][t - td_wl - td_syn - 1] - I_ps_M[1][t - td_wl - td_syn - 1]
        
            v_M[m][t] = v_M_aux + dt*(dv_M + coupling_cortex + coupling_thalamus + bg_activity + M_fid*dbs_I)
            u_M[m][t] = u_M_aux + dt*du_M
            
        # Synapse - Within cortex  
        syn_M = tm_synapse_eq(u = u_M_syn, R = R_M_syn, I = I_M_syn, AP = AP_M_aux, t_f = t_f_E, t_d = t_d_E, t_s = t_s_E, U = U_E, A = A_E, dt = dt, p = p)
        
        R_M_syn = 1*syn_M['R']
        u_M_syn = 1*syn_M['u']
        I_M_syn = 1*syn_M['I']
        I_syn[0][m] = 1*syn_M['Ipost']
        
    PSC_M[0][t] = np.sum(I_syn)
    return v_M, u_M, PSC_M, u_M_syn, I_M_syn, R_M_syn