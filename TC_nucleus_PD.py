"""
Created on Sat Feb  3 19:21:12 2024

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

n_TC = neuron_quantities['TC']

vr = TCM_model_parameters()['vr']
vp = TCM_model_parameters()['vp']

W_PD = coupling_matrix_PD()['weights']

W_TC_self = W_PD['W_EE_tc']
W_TC_S = W_PD['W_EE_tc_s']
W_TC_M = W_PD['W_EE_tc_m']
W_TC_D = W_PD['W_EE_tc_d']
W_TC_TR = W_PD['W_EI_tc_tr']
W_TC_CI = W_PD['W_EI_tc_ci']

td_wl = TCM_model_parameters()['time_delay_within_layers']
td_bl = TCM_model_parameters()['time_delay_between_layers']
td_ct = TCM_model_parameters()['time_delay_cortex_thalamus']
td_tc = TCM_model_parameters()['time_delay_thalamus_cortex']
td_syn = TCM_model_parameters()['time_delay_synapse']
p = TCM_model_parameters()['synapse_total_params']

a_TC = neuron_params['a_TC']
b_TC = neuron_params['b_TC']
c_TC = neuron_params['c_TC']
d_TC = neuron_params['d_TC']

t_f_E = syn_params['t_f']
t_d_E = syn_params['t_d']
t_s_E = syn_params['t_s']
U_E = syn_params['U']
A_E = syn_params['distribution']
A_E_T_D = syn_params['distribution_T_D']

I_TC = currents['TC']

noise = TCM_model_parameters()['noise']

kisi_TC = noise['kisi_TC']
zeta_TC = noise['zeta_TC']

I_ps = TCM_model_parameters()['poisson_bg_activity']
I_ps_TC = I_ps['TC']

def TC_nucleus(t, v_TC, u_TC, AP_TC, PSC_TC, PSC_S, PSC_M, PSC_D_T, PSC_TR, PSC_CI, PSC_T_D, R_TC_syn, u_TC_syn, I_TC_syn):
    
    I_syn = np.zeros((1, n_TC))
    I_syn_t = np.zeros((1, n_TC))
    
    for tc in range(n_TC):
        v_TC_aux = 1*v_TC[tc][t - 1]
        u_TC_aux = 1*u_TC[tc][t - 1]
        AP_TC_aux = 0
                
        if (v_TC_aux >= vp + zeta_TC[tc][t - 1]):
            AP_TC_aux = 1
            AP_TC[tc][t] = t - 1
            v_TC_aux = v_TC[tc][t]
            v_TC[tc][t] = c_TC[0][tc]
            u_TC[tc][t] = u_TC_aux + d_TC[0][tc]
        else:
            AP_TC[tc][t] = 0
            AP_TC_aux = 0
            
            # Self feedback - Inhibitory
            coupling_TC_TC = W_TC_self[tc][0]*1*PSC_TC[0][t - td_wl - td_syn - 1]
            # Coupling TC to S - Excitatory 
            coupling_TC_S = W_TC_S[tc][0]*1*PSC_S[0][t - td_ct - td_syn - 1]
            # Coupling TC to M - Excitatory 
            coupling_TC_M = W_TC_M[tc][0]*1*PSC_M[0][t - td_ct - td_syn - 1]
            # Coupling TC to D - Excitatory 
            coupling_TC_D = W_TC_D[tc][0]*1*PSC_D_T[0][t - td_ct - td_syn - 1]
            # Coupling TC to CI - Inhibitory 
            coupling_TC_CI = W_TC_CI[tc][0]*1*PSC_CI[0][t - td_ct - td_syn - 1]
            # Coupling TC to TR - Excitatory 
            coupling_TC_TR = W_TC_TR[tc][0]*1*PSC_TR[0][t - td_bl - td_syn - 1]
            
            dv_TC = izhikevich_dvdt(v = v_TC_aux, u = u_TC_aux, I = I_TC[tc])
            du_TC = izhikevich_dudt(v = v_TC_aux, u = u_TC_aux, a = a_TC[0][tc], b = b_TC[0][tc])
            
            coupling_cortex = (coupling_TC_S + coupling_TC_M + coupling_TC_D + coupling_TC_CI)/n_TC
            coupling_thalamus = (coupling_TC_TC + coupling_TC_TR)/n_TC
            bg_activity = kisi_TC[tc][t - 1] + I_ps_TC[0][t - td_wl - td_syn - 1] - I_ps_TC[1][t - td_wl - td_syn - 1]
        
            v_TC[tc][t] = v_TC_aux + dt*(dv_TC + coupling_cortex + coupling_thalamus + bg_activity)
            u_TC[tc][t] = u_TC_aux + dt*du_TC
            
        u = 1*u_TC_syn
        R = 1*R_TC_syn
        I = 1*I_TC_syn
        # Synapse - Within layer  
        syn_TC = tm_synapse_eq(u = u_TC_syn, 
                              R = R_TC_syn, 
                              I = I_TC_syn, 
                              AP = AP_TC_aux, 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E, 
                              dt = dt, 
                              p = p)
        
        # Synapse - With cortex
        syn_TC_D = tm_synapse_eq(u = u, 
                              R = R, 
                              I = I, 
                              AP = AP_TC_aux, 
                              t_f = t_f_E, 
                              t_d = t_d_E, 
                              t_s = t_s_E, 
                              U = U_E, 
                              A = A_E_T_D, 
                              dt = dt, 
                              p = p)
        
        R_TC_syn = 1*syn_TC['R']
        u_TC_syn = 1*syn_TC['u']
        I_TC_syn = 1*syn_TC['I']
        I_syn[0][tc] = 1*syn_TC['Ipost']
        
        I_syn_t[0][tc] = 1*syn_TC_D['Ipost']
        
    PSC_TC[0][t] = np.sum(I_syn)
    PSC_T_D[0][t] = np.sum(I_syn_t)
    
    return v_TC, u_TC, PSC_TC, u_TC_syn, I_TC_syn, R_TC_syn, PSC_T_D