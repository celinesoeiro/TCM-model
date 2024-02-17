"""
@author: Celine Soeiro

@description: Thalamo-Cortical microcircuit by AmirAli Farokhniaee and Madeleine M. Lowery - 2021
This info was found in the IEEE conference paper provided by the authors

# Abreviations:
    PD: Parkinson Desease
    S: Superficial layer
    M: Medium layer
    D: Deep layer
    CI: Cortical Interneurons
    TC: Thalamo-Cortical Relay Nucleus (TC)
    TR: Thalamic Reticular Nucleus (TR)
    PD: Poissonian Distribution 
    DBS: Deep Brain Stimulation
    
    This model consists of populations of excitatory and inhibitory point-like spiking neurns in the motor cortex
and thalamus.
    The excitatory neurons in the motor cortex were divided into 3 layers of pyramidal neurons (PN), surface (S),
middle (M) and deep (D).
    The inhibitory neurons in the motor cortex were considered as a single population of cortical interneurons (CI).
    The excitatory neurons in the thalamus formed the thalamocortical relay nucleus (TC) and the inhibitory neurons
comprised the thalamic retcular nucleus (TR).

# NEURONS PER STRUCTURE
S: Excitatory
    - Regular Spiking (RS)
    - Intrinsically Bursting (IB)
M: Excitatory
    - Regular Spiking (RS)
D: Excitatory
    - Regular Spiking (RS)
    - Intrinsically Bursting (IB)
CI: Inhibitory
    - Fast spiking (FS)
    - Low Threshold Spiking (LTS)
TC: Excitatory
    - Thalamocortical (TC)
TR: Inhibitory
    - Thalamic Reticular (TR)
    
# SYNAPTIC INPUTS
    Connections between the neurons in the network modal were considered as a 
    combination of Facilitating (F), Depressing (D) and Pseudo-Linear (P) 
    synapses with distribution:
    F: 8%
    D: 75%
    P: 15%
    Connection between layer D and Thamalus -> Pure Facilitating
    Connection between TCR and Layer D -> Pure Depressing
    
# NETWORK CONNECTIONS

"""
import random
import numpy as np

from model_functions import poisson_spike_generator, tm_synapse_poisson_eq

def TCM_model_parameters():
    random.seed(0)
    random_factor = np.round(random.random(),2)
    
    ms = 1000                               # 1 second = 1000 miliseconds
    dt = 100/ms                             # time step of 10 ms
    simulation_time = 15                    # simulation time in seconds
    samp_freq = int(ms/dt)                  # sampling frequency in Hz
    T = int((simulation_time)*ms)           # Simulation time in ms with 1 extra second to reach the steady state and trash later
    sim_steps = int(T/dt)                   # number of simulation steps
    chop_till = 1*samp_freq                 # Cut the first 1 seconds of the simulation

    td_synapse = 1                          # Synaptic transmission delay (fixed for all synapses in the TCM)
    td_thalamus_cortex = 15                 # time delay from thalamus to cortex (ms) (transmission time delay)
    td_cortex_thalamus = 20                 # time delay from cortex to thalamus (ms) (transmission time delay)  
    td_layers = 8                           # time delay between the layers in cortex and nuclei in thalamus (ms) (PSC delay)
    td_within_layers = 1                    # time delay within a structure (ms)
    
    # Time vector
    if (td_thalamus_cortex >= td_cortex_thalamus): 
        t_vec = np.arange(td_thalamus_cortex + td_synapse, sim_steps)
    else:
        t_vec = np.arange(td_cortex_thalamus + td_synapse, sim_steps)

    Idc_tune = 0.1                          # 
    vr = -65                                # membrane potential resting value 
    vp = 30                                 # membrane peak voltage value
        
    hyperdirect_neurons = 0.1               # percentage of PNs that are hyperdirect -> For DBS
    
    connectivity_factor_normal = 2.5        # For 100 neurons
    connectivity_factor_PD = 5              # For 100 neurons
        
    dbs_on = int(5*67)                      # value of synaptic fidelity when DBS on
    dbs_off = 0                             # value of synaptic fidelity when DBS off
    dbs_amplitude = 1                       # 1 uA
    dbs_freq = 80
        
    # Neuron quantities
    qnt_neurons_s = 10         # Excitatory
    qnt_neurons_m = 10         # Excitatory
    qnt_neurons_d = 10         # Excitatory
    qnt_neurons_ci = 10        # Inhibitory
    qnt_neurons_tc = 10        # Excitatory
    qnt_neurons_tr = 4         # Inhibitory
    
    neuron_quantities = {
        'S': qnt_neurons_s,                      # Number of neurons in Superficial layer
        'M': qnt_neurons_m,                      # Number of neurons in Medium layer
        'D': qnt_neurons_d,                      # Number of neurons in Deep layer
        'CI': qnt_neurons_ci,                    # Number of IC neurons
        'TC': qnt_neurons_tc,                    # Number of neurons in TC
        'TR': qnt_neurons_tr,                    # Number of neurons in TR
        'HD': qnt_neurons_d*hyperdirect_neurons, # Number of hyperdirect neurons
        'total': qnt_neurons_s + qnt_neurons_m + qnt_neurons_d + qnt_neurons_ci + qnt_neurons_tc + qnt_neurons_tr,
        }
    
    # Distribution of neurons in each structure
    neurons_s_1 = int(0.5*qnt_neurons_s)        # RS neurons
    neurons_s_2 = int(0.5*qnt_neurons_s)        # IB neurons
    neurons_m_1 = int(1*qnt_neurons_m)          # RS neurons
    neurons_m_2 = int(0*qnt_neurons_m)          # IB neurons
    neurons_d_1 = int(0.7*qnt_neurons_d)        # RS neurons
    neurons_d_2 = int(0.3*qnt_neurons_d)        # IB neurons
    neurons_ci_1 = int(0.5*qnt_neurons_ci)      # FS neurons
    neurons_ci_2 = int(0.5*qnt_neurons_ci)      # LTS neurons
    neurons_tr_1 = int(0.5*qnt_neurons_tr)      # TR neurons
    neurons_tr_2 = int(0.5*qnt_neurons_tr)      # TR neurons
    neurons_tc_1 = int(0.7*qnt_neurons_tc)      # TC neurons
    neurons_tc_2 = int(0.3*qnt_neurons_tc)      # TC neurons
    
    neuron_types_S = ['RS']*neurons_s_1 + ['IB']*neurons_s_2
    neuron_types_M = ['RS']*neurons_m_1 + ['IB']*neurons_m_2
    neuron_types_D = ['RS']*neurons_d_1 + ['IB']*neurons_d_2
    neuron_types_CI = ['FS']*neurons_ci_1 + ['LTS']*neurons_ci_2
    neuron_types_TC = ['TC']*neurons_ci_1 + ['TC']*neurons_ci_2
    neuron_types_TR = ['TR']*neurons_ci_1 + ['TR']*neurons_ci_2
    
    neuron_per_structure = {
        'neurons_s_1': neurons_s_1,             # Regular Spiking
        'neurons_s_2': neurons_s_2,             # Intrinsically Bursting
        'neurons_m_1': neurons_m_1,             # Regular Spiking
        'neurons_m_2': neurons_m_2,             # Regular Spiking
        'neurons_d_1': neurons_d_1,             # Regular Spiking
        'neurons_d_2': neurons_d_2,             # Intrinsically bursting
        'neurons_ci_1': neurons_ci_1,           # Fast spiking
        'neurons_ci_2': neurons_ci_2,           # Low threshold spiking
        'neurons_tc_1': neurons_tc_1,           # Reley
        'neurons_tc_2': neurons_tc_2,           # Relay
        'neurons_tr_1': neurons_tr_1,           # Reticular
        'neurons_tr_2': neurons_tr_2,           # Reticular
        }
    
    neuron_types_per_structure = {
        'S': neuron_types_S,
        'M': neuron_types_M,
        'D': neuron_types_D,
        'CI': neuron_types_CI,
        'TC': neuron_types_TC,
        'TR': neuron_types_TR
        }
    
    # Neuron parameters to model Izhikevich Neurons
    # 0 - RS - Regular Spiking
    # 1 - IB - Intrinsically Bursting
    # 2 - FS - Fast Spiking
    # 3 - LTS - Low Threshold Spiking
    # 4 - TC (rel) - Thalamo-Cortical Relay
    # 5 - TR - Thalamic Reticular
    
    # =============================================================================
    #     Making neuron variations (all neurons must be different from one another)
    # =============================================================================

    #    0-RS  1-IB  2-FS 3-LTS 4-TC  5-TR 
    a = [0.02, 0.02, 0.1, 0.02, 0.02, 0.02]
    b = [0.2,  0.2,  0.2, 0.25, 0.25, 0.25]
    c = [-65,  -55,  -65, -65,   -65,  -65]
    d = [8,    4,      2,   2,  0.05, 2.05]
    
    r_s_1 = np.random.rand(1,neurons_s_1); r_s_2 = np.random.rand(1,neurons_s_2);
    
    a_S = np.c_[a[0]*np.ones((1, neurons_s_1)), a[1]*np.ones((1, neurons_s_2))]
    b_S = np.c_[b[0]*np.ones((1, neurons_s_1)), b[1]*np.ones((1, neurons_s_2))]
    c_S = np.c_[c[0] + 15*r_s_1**2, c[1] + 15*r_s_2**2]
    d_S = np.c_[d[0] - 0.6*r_s_1**2, d[1] -0.6*r_s_2**2]
    
    r_m_1 = np.random.rand(1,neurons_m_1); r_m_2 = np.random.rand(1,neurons_m_2);
    
    a_M = np.c_[a[0]*np.ones((1, neurons_m_1)), a[0]*np.ones((1, neurons_m_2))]
    b_M = np.c_[b[0]*np.ones((1, neurons_m_1)), b[0]*np.ones((1, neurons_m_2))]
    c_M = np.c_[c[0] + 15*r_m_1**2, c[0] + 15*r_m_2**2]
    d_M = np.c_[d[0] - 0.6*r_m_1**2, d[0] -0.6*r_m_2**1]
    
    r_d_1 = np.random.rand(1,neurons_d_1); r_d_2 = np.random.rand(1,neurons_d_2);
    
    a_D = np.c_[a[0]*np.ones((1, neurons_d_1)), a[1]*np.ones((1, neurons_d_2))]
    b_D = np.c_[b[0]*np.ones((1, neurons_d_1)), b[1]*np.ones((1, neurons_d_2))]
    c_D = np.c_[c[0] + 15*r_d_1**2, c[1] + 15*r_d_2**2]
    d_D = np.c_[d[0] - 0.6*r_d_1**2, d[1] - 0.6*r_d_2**2]
        
    r_ci_1 = np.random.rand(1,neurons_ci_1); r_ci_2 = np.random.rand(1,neurons_ci_2);
    
    a_CI = np.c_[a[2] + 0.008*r_ci_1, a[3] + 0.008*r_ci_2]
    b_CI = np.c_[b[2] - 0.005*r_ci_1, b[3] - 0.005*r_ci_2]
    c_CI = np.c_[c[2]*np.ones((1, neurons_ci_1)), c[3]*np.ones((1, neurons_ci_2))]
    d_CI = np.c_[d[2]*np.ones((1, neurons_ci_1)), d[3]*np.ones((1, neurons_ci_2))]
    
    r_tc_1 = np.random.rand(1,neurons_tc_1); r_tc_2 = np.random.rand(1,neurons_tc_2);
    
    a_TC = np.c_[a[4] + 0.008*r_tc_1, a[4] + 0.008*r_tc_2]
    b_TC = np.c_[b[4] - 0.005*r_tc_1, b[4] - 0.005*r_tc_2]
    c_TC = np.c_[c[4]*np.ones((1, neurons_tc_1)), c[4]*np.ones((1, neurons_tc_2))] 
    d_TC = np.c_[d[4]*np.ones((1, neurons_tc_1)), d[4]*np.ones((1, neurons_tc_2))] 
    
    r_tr_1 = np.random.rand(1,neurons_tr_1); r_tr_2 = np.random.rand(1,neurons_tr_2);
    
    a_TR = np.c_[a[5] + 0.008*r_tr_1, a[5] + 0.008*r_tr_2]
    b_TR = np.c_[b[5] - 0.005*r_tr_1, b[5] - 0.005*r_tr_2]
    c_TR = np.c_[c[5]*np.ones((1, neurons_tr_1)), c[5]*np.ones((1, neurons_tr_2))]
    d_TR = np.c_[d[5]*np.ones((1, neurons_tr_1)), d[5]*np.ones((1, neurons_tr_2))]
        
    neuron_params = {
        'a_S': a_S,
        'b_S': b_S,
        'c_S': c_S,
        'd_S': d_S,
        'a_M': a_M,
        'b_M': b_M, 
        'c_M': c_M,
        'd_M': d_M,
        'a_D': a_D,
        'b_D': b_D,
        'c_D': c_D,
        'd_D': d_D,
        'a_CI': a_CI,
        'b_CI': b_CI,
        'c_CI': c_CI,
        'd_CI': d_CI,
        'a_TR': a_TR,
        'b_TR': b_TR,
        'c_TR': c_TR,
        'd_TR': d_TR,
        'a_TC': a_TC,
        'b_TC': b_TC,
        'c_TC': c_TC,
        'd_TC': d_TC,
        }

    # =============================================================================
    #     Noise terms
    # =============================================================================
    white_gaussian_add = 1.5; cn = 1; # additive white Gaussian noise strength
    white_gaussian_thr = 0.5 # threshold white Gaussian noise strength

    random_S = np.random.randn(qnt_neurons_s, samp_freq)
    random_M = np.random.randn(qnt_neurons_m, samp_freq)
    random_D = np.random.randn(qnt_neurons_d, samp_freq)
    random_CI = np.random.randn(qnt_neurons_ci, samp_freq)
    random_TR = np.random.randn(qnt_neurons_tr, samp_freq)
    random_TC = np.random.randn(qnt_neurons_tc, samp_freq)
    
    random_S_diff = np.random.randn(qnt_neurons_s, sim_steps - samp_freq)
    random_M_diff = np.random.randn(qnt_neurons_m, sim_steps - samp_freq)
    random_D_diff = np.random.randn(qnt_neurons_d, sim_steps - samp_freq)
    random_CI_diff = np.random.randn(qnt_neurons_ci, sim_steps - samp_freq)
    random_TR_diff = np.random.randn(qnt_neurons_tr, sim_steps - samp_freq)
    random_TC_diff = np.random.randn(qnt_neurons_tc, sim_steps - samp_freq)

    zeta_S_E = white_gaussian_thr*np.c_[ random_S, cn*random_S_diff ]
    zeta_M_E = white_gaussian_thr*np.c_[ random_M, cn*random_M_diff ]    
    zeta_D_E = white_gaussian_thr*np.c_[random_D, cn*random_D_diff ]
    zeta_CI_I = white_gaussian_thr*np.c_[random_CI, cn*random_CI_diff ]
    zeta_TR_I = white_gaussian_thr*np.c_[random_TR, cn*random_TR_diff ]
    zeta_TC_E = white_gaussian_thr*np.c_[random_TC, cn*random_TC_diff ]
    
    kisi_S_E = white_gaussian_add*np.c_[ random_S, cn*random_S_diff ]
    kisi_M_E = white_gaussian_add*np.c_[ random_M, cn*random_M_diff ]    
    kisi_D_E = white_gaussian_add*np.c_[random_D, cn*random_D_diff ]
    kisi_CI_I = white_gaussian_add*np.c_[ random_CI, cn*random_CI_diff ]
    kisi_TC_E = white_gaussian_add*np.c_[ random_TC, cn*random_TC_diff ]
    kisi_TR_I = white_gaussian_add*np.c_[ random_TR, cn*random_TR_diff ]
    
    noise = {
        'kisi_S': kisi_S_E,
        'kisi_M': kisi_M_E,
        'kisi_D': kisi_D_E,
        'kisi_CI': kisi_CI_I,
        'kisi_TC': kisi_TC_E,
        'kisi_TR': kisi_TR_I,
        'zeta_S': zeta_S_E,
        'zeta_M': zeta_M_E,
        'zeta_D': zeta_D_E,
        'zeta_CI': zeta_CI_I,
        'zeta_TC': zeta_TC_E,
        'zeta_TR': zeta_TR_I,
        }
    
    # Bias currents (Subthreshold CTX and Suprethreshold THM) - Will be used in the neurons
    Idc = [3.6, 3.7, 3.9, 0.5, 0.7]
    
    I_S_1 = Idc[0]
    I_S_2 = Idc[1]
    I_M_1 = Idc[0]
    I_M_2 = Idc[0]
    I_D_1 = Idc[0]
    I_D_2 = Idc[1]
    I_CI_1 = Idc[2]
    I_CI_2 = Idc[3]
    I_TR_1 = Idc[4]
    I_TR_2 = Idc[4]
    I_TC_1 = Idc[4]
    I_TC_2 = Idc[4]
    
    I_S = np.concatenate((I_S_1*np.ones((1, neurons_s_1)), I_S_2*np.ones((1, neurons_s_2))), axis=None)
    I_M = np.concatenate((I_M_1*np.ones((1, neurons_m_1)), I_M_2*np.ones((1, neurons_m_2))), axis=None)
    I_D = np.concatenate((I_D_1*np.ones((1, neurons_d_1)), I_D_2*np.ones((1, neurons_d_2))), axis=None)
    I_CI = np.concatenate((I_CI_1*np.ones((1, neurons_ci_1)), I_CI_2*np.ones((1, neurons_ci_2))), axis=None)
    I_TR = np.concatenate((I_TR_1*np.ones((1, neurons_tr_1)), I_TR_2*np.ones((1, neurons_tr_2))), axis=None)
    I_TC = np.concatenate((I_TC_1*np.ones((1, neurons_tc_1)), I_TC_2*np.ones((1, neurons_tc_2))), axis=None)
    
    currents_per_structure = {
        'S': I_S,
        'M': I_M,
        'D': I_D,
        'CI': I_CI,
        'TR': I_TR,
        'TC': I_TC,
        }
    
    # =============================================================================
    #     SYNAPSE INITIAL VALUES
    # =============================================================================
    p = 3
    synapse_params_excitatory = {
        't_f': [670, 17, 326],
        't_d': [138, 671, 329],
        'U': [0.09, 0.5, 0.29],
        'distribution': [0.2, 0.63, 0.17],
        'distribution_T_D': [0, 1, 0], # Depressing
        'distribution_D_T': [1, 0, 0], # Facilitating
        't_s': 3,
        }
    
    synapse_params_inhibitory = {
        't_f': [376, 21, 62],
        't_d': [45, 706, 144],
        'U': [0.016, 0.25, 0.32],
        'distribution': [0.08, 0.75, 0.17],
        't_s': 11,
        }
    # =============================================================================
    # POISSONIAN BACKGROUND ACTIVITY 
    # - Poissonian postsynaptic input to the E and I neurons for all layers
    # =============================================================================
    w_ps = 1
    I_ps_S = np.zeros((2, sim_steps))
    I_ps_M = np.zeros((2, sim_steps))
    I_ps_D = np.zeros((2, sim_steps))
    I_ps_CI = np.zeros((2, sim_steps))
    I_ps_TR = np.zeros((2, sim_steps))
    I_ps_TC = np.zeros((2, sim_steps))
    ps_firing_rates = np.zeros((1,6))
    
    for i in range(6):
        ps_firing_rates[0][i] = 20 + 2 * np.random.randn()

    W_ps = [[w_ps * np.random.randn() for _ in range(2)] for _ in range(6)]

    def bg_currents(frequency):
        [spike_PS, I_PS] = poisson_spike_generator(num_steps = sim_steps, 
                                                dt = dt, 
                                                num_neurons = 1, 
                                                thalamic_firing_rate = frequency, 
                                                current_value=None)
    
        # Mudar I_E e I_I para gerar um array e colocar o I_PS_x para receber esse array * o peso
        I_E = tm_synapse_poisson_eq(spikes = spike_PS,  
                                    sim_steps = sim_steps, 
                                    dt = dt, 
                                    t_f = synapse_params_excitatory['t_f'], 
                                    t_d = synapse_params_excitatory['t_d'], 
                                    t_s = synapse_params_excitatory['t_s'], 
                                    U = synapse_params_excitatory['U'], 
                                    A = synapse_params_excitatory['distribution'], 
                                    time = t_vec)
    
        I_I = tm_synapse_poisson_eq(spikes = spike_PS, 
                                    sim_steps = sim_steps, 
                                    dt = dt, 
                                    t_f = synapse_params_inhibitory['t_f'], 
                                    t_d = synapse_params_inhibitory['t_d'], 
                                    t_s = synapse_params_inhibitory['t_s'], 
                                    U = synapse_params_inhibitory['U'], 
                                    A = synapse_params_inhibitory['distribution'],
                                    time = t_vec)
        return I_E, I_I
    
    bg_S_E, bg_S_I = bg_currents(ps_firing_rates[0][0])
    bg_M_E, bg_M_I = bg_currents(ps_firing_rates[0][1])
    bg_D_E, bg_D_I = bg_currents(ps_firing_rates[0][2])
    bg_CI_E, bg_CI_I = bg_currents(ps_firing_rates[0][3])
    bg_TC_E, bg_TC_I = bg_currents(ps_firing_rates[0][4])
    bg_TR_E, bg_TR_I = bg_currents(ps_firing_rates[0][5])
    
    # Excitatory
    I_ps_S[0] = W_ps[0][0]*bg_S_E
    I_ps_M[0] = W_ps[1][0]*bg_M_E
    I_ps_D[0] = W_ps[2][0]*bg_D_E
    I_ps_CI[0] = W_ps[3][0]*bg_CI_E
    I_ps_TR[0] = W_ps[4][0]*bg_TR_E
    I_ps_TC[0] = W_ps[5][0]*bg_TC_E

    # Inhibitory
    I_ps_S[1] = W_ps[0][1]*bg_S_I
    I_ps_M[1] = W_ps[1][1]*bg_M_I
    I_ps_D[1] = W_ps[2][1]*bg_D_I
    I_ps_CI[1] = W_ps[3][1]*bg_CI_I
    I_ps_TR[1] = W_ps[4][1]*bg_TR_I
    I_ps_TC[1] = W_ps[5][1]*bg_TC_I
    
    I_ps = {
        'S': I_ps_S,
        'M': I_ps_M,
        'D': I_ps_D,
        'CI': I_ps_CI,
        'TC': I_ps_TC,
        'TR': I_ps_TR,
        }
    
    # =============================================================================
    #     DBS
    # =============================================================================
    # synaptic_fidelity = 0 # DBS off
    synaptic_fidelity = 5*67 # DBS on
    
    fid_CI = np.abs(1*synaptic_fidelity)
    fid_S = np.abs(1*synaptic_fidelity)
    fid_M = np.abs(0*synaptic_fidelity)
    fid_D = np.abs(1*synaptic_fidelity)
    fid_TC = np.abs(1*synaptic_fidelity)
    fid_TR = np.abs(1*synaptic_fidelity)
    
    nS = 1; nM = 0; nCI = 1; nTC = 1; nTR = 1;
    # Percentage of neurons that have synaptic contact with hyperdirect neurons axon arbors
    neurons_connected_with_hyperdirect_neurons = {
        'S': nS*hyperdirect_neurons*qnt_neurons_s,   # percentage of S neurons that have synaptic contact with hyperdirect neurons axon arbors
        'M': nM*hyperdirect_neurons*qnt_neurons_m,   # percentage of M neurons that have synaptic contact with hyperdirect neurons axon arbors
        'D': hyperdirect_neurons*qnt_neurons_d,
        'CI': nCI*hyperdirect_neurons*qnt_neurons_ci,# percentage of CI neurons that have synaptic contact with hyperdirect neurons axon arbors
        'TR': nTR*hyperdirect_neurons*qnt_neurons_tr, # percentage of R neurons that have synaptic contact with hyperdirect neurons axon arbors
        'TC': nTC*hyperdirect_neurons*qnt_neurons_tc, # percentage of N neurons that have synaptic contact with hyperdirect neurons axon arbors
        }
    
    synaptic_fidelity_layers = {
        'S': fid_S,
        'M': fid_M,
        'D': fid_D,
        'CI': fid_CI,
        'TC': fid_TC,
        'TR': fid_TR,
        }
    
    # Export all dictionaries
    data = {
        'hyperdirect_neurons': hyperdirect_neurons, # Percentage of PNs affected in D by DBS
        'simulation_time': simulation_time, # simulation time in seconds (must be a multiplacative of 3 under PD+DBS condition)
        'simulation_time_ms': T,
        'dt': dt, # time step
        'sampling_frequency': samp_freq, # in Hz
        'simulation_steps': sim_steps,
        'chop_till': chop_till, # cut the first 1s of simulation
        'time_delay_between_layers': td_layers,
        'time_delay_within_layers': td_within_layers,
        'time_delay_thalamus_cortex': td_thalamus_cortex,
        'time_delay_cortex_thalamus': td_cortex_thalamus,
        'time_delay_synapse': td_synapse,
        'time_vector': t_vec,
        'connectivity_factor_normal_condition': connectivity_factor_normal,
        'connectivity_factor_PD_condition': connectivity_factor_PD,
        'vr': vr,
        'vp': vp,
        'Idc_tune': Idc_tune,
        'neuron_types_per_structure': neuron_types_per_structure,
        'neuron_quantities': neuron_quantities,
        'neuron_per_structure': neuron_per_structure,
        'neurons_connected_with_hyperdirect_neurons': neurons_connected_with_hyperdirect_neurons,
        'neuron_paramaters': neuron_params,
        'bias_current': Idc,
        'currents_per_structure': currents_per_structure,
        'noise': noise,
        'random_factor': random_factor,
        'synapse_params_excitatory': synapse_params_excitatory,
        'synapse_params_inhibitory': synapse_params_inhibitory,
        'synapse_total_params': p,
        'dbs': [dbs_off, dbs_on],
        'poisson_bg_activity': I_ps,
        'synaptic_fidelity_layers': synaptic_fidelity_layers,
        'dbs_amplitude': dbs_amplitude,
        'dbs_freq': dbs_freq
        }
    
    return data

def coupling_matrix_normal():
    neuron_quantities = TCM_model_parameters()['neuron_quantities']
    facilitating_factor = TCM_model_parameters()['connectivity_factor_normal_condition']
    
    n_s = neuron_quantities['S']
    n_m = neuron_quantities['M']
    n_d = neuron_quantities['D']
    n_ci = neuron_quantities['CI']
    n_tc = neuron_quantities['TC']
    n_tr = neuron_quantities['TR']

    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s,1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ci = initial + interval*np.random.rand(n_ci, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory 
    # II -> Inhibitory to Inhibitory 
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -1e1/facilitating_factor;            W_EE_s = aee_s*r_s;
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -1e1/facilitating_factor;            W_EE_m = aee_m*r_m;
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -1e1/facilitating_factor;            W_EE_d = aee_d*r_d;
    ## INs 
    aii_ci = -5e2/facilitating_factor;          W_II_ci = aii_ci*r_ci;
    ## Reticular cells
    aii_tr = -5e1/facilitating_factor;          W_II_tr = aii_tr*r_tr;
    ## Relay cells
    aee_tc = 0/facilitating_factor;             W_EE_tc = aee_tc*r_tc;
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 1e1/facilitating_factor;          W_EE_s_m = aee_sm*r_s;
    # D to S coupling
    aee_sd = 5e2/facilitating_factor;          W_EE_s_d = aee_sd*r_s;
    # CI to S coupling
    aei_sci = -5e2/facilitating_factor;        W_EI_s_ci = aei_sci*r_s;
    # Reticular to S coupling
    aei_str = 0/facilitating_factor;           W_EI_s_tr = aei_str*r_s;
    # Rel. to S couplings
    aee_stc = 0/facilitating_factor;           W_EE_s_tc = aee_stc*r_s;     
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 3e2/facilitating_factor;       W_EE_m_s = aee_ms*r_m; 
    # D to M couplings
    aee_md = 0/facilitating_factor;         W_EE_m_d = aee_md*r_m;            
    # CI to M couplings
    aei_mci = -3e2/facilitating_factor;     W_EI_m_ci = aei_mci*r_m;
    # Ret. to M couplings    
    aei_mtr = 0/facilitating_factor;        W_EI_m_tr = aei_mtr*r_m;
    # Rel. to M couplings
    aee_mtc = 0/facilitating_factor;        W_EE_m_tc = aee_mtc*r_m;
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor;       W_EE_d_s = aee_ds*r_d;
    # M to D couplings
    aee_dm = 0/facilitating_factor;         W_EE_d_m = aee_dm*r_d;
    # CI to D couplings
    aei_dci = -7.5e3/facilitating_factor;   W_EI_d_ci = aei_dci*r_d;
    # Ret. to D couplings
    aei_dtr = 0/facilitating_factor;        W_EI_d_tr = aei_dtr*r_d;
    # Rel. to D couplings
    aee_dtc = 1e1/facilitating_factor;      W_EE_d_tc = aee_dtc*r_d;
    # =============================================================================
    #     CI
    # =============================================================================
    # S to CIs couplings
    aie_CIs = 2e2/facilitating_factor;     W_IE_ci_s = aie_CIs*r_ci;
    # M to CIs couplings
    aie_CIm = 2e2/facilitating_factor;     W_IE_ci_m = aie_CIm*r_ci;
    # D to CIs couplings
    aie_CId = 2e2/facilitating_factor;     W_IE_ci_d = aie_CId*r_ci;
    # Ret. to CIs couplings
    aii_CITR = 0/facilitating_factor;      W_II_ci_tr = aii_CITR*r_ci;
    # Rel. to CIs couplings
    aie_CITC = 1e1/facilitating_factor;    W_IE_ci_tc = aie_CITC*r_ci;
    # =============================================================================
    #     TR
    # =============================================================================
    # S to Ret couplings
    aie_trs = 0/facilitating_factor;       W_IE_tr_s = aie_trs*r_tr;
    # M to Ret couplings
    aie_trm = 0/facilitating_factor;       W_IE_tr_m = aie_trm*r_tr;
    # D to Ret couplings
    aie_trd = 7e2/facilitating_factor;     W_IE_tr_d = aie_trd*r_tr;
    # CI to Ret couplings
    aii_trci = 0/facilitating_factor;      W_II_tr_ci = aii_trci*r_tr;
    # Rel. to Ret couplings
    aie_trtc = 1e3/facilitating_factor;    W_IE_tr_tc = aie_trtc*r_tr;
    # =============================================================================
    #     TC
    # =============================================================================
    # S to Rel couplings
    aee_tcs = 0/facilitating_factor;       W_EE_tc_s = aee_tcs*r_tc;   
    # M to Rel couplings
    aee_tcm = 0/facilitating_factor;       W_EE_tc_m = aee_tcm*r_tc;
    # D to Rel couplings
    aee_tcd = 7e2/facilitating_factor;     W_EE_tc_d = aee_tcd*r_tc;
    # CI to Rel couplings
    aei_tcci = 0/facilitating_factor;      W_EI_tc_ci = aei_tcci*r_tc;
    # Ret to Rel couplings
    aei_tctr = -5e2/facilitating_factor;   W_EI_tc_tr = aei_tctr*r_tc;
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # 0 -> Layer S
    # 1 -> Layer M
    # 2 -> Layer D
    # 3 -> CI
    # 4 -> TR
    # 5 -> TC
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_EE_tc)
    matrix[5][5] = np.mean(W_II_tr)
    # First column - Layer S
    matrix[1][0] = np.mean(W_EE_s_m)
    matrix[2][0] = np.mean(W_EE_s_d)
    matrix[3][0] = np.mean(W_EI_s_ci)
    matrix[4][0] = np.mean(W_EE_s_tc)
    matrix[5][0] = np.mean(W_EI_s_tr)
    # Second column - Layer M
    matrix[0][1] = np.mean(W_EE_m_s)
    matrix[2][1] = np.mean(W_EE_m_d)
    matrix[3][1] = np.mean(W_EI_m_ci)
    matrix[4][1] = np.mean(W_EE_m_tc)
    matrix[5][1] = np.mean(W_EI_m_tr)
    # Thid column - Layer D
    matrix[0][2] = np.mean(W_EE_d_s)
    matrix[1][2] = np.mean(W_EE_d_m)
    matrix[3][2] = np.mean(W_EI_d_ci)
    matrix[4][2] = np.mean(W_EE_d_tc)
    matrix[5][2] = np.mean(W_EI_d_tr)
    # Fourth column - Structure CI
    matrix[0][3] = np.mean(W_IE_ci_s)
    matrix[1][3] = np.mean(W_IE_ci_m)
    matrix[2][3] = np.mean(W_IE_ci_d)
    matrix[4][3] = np.mean(W_IE_ci_tc)
    matrix[5][3] = np.mean(W_II_ci_tr)
    # Fifth column - Structure TC
    matrix[0][4] = np.mean(W_EE_tc_s)
    matrix[1][4] = np.mean(W_EE_tc_m)
    matrix[2][4] = np.mean(W_EE_tc_d)
    matrix[3][4] = np.mean(W_EI_tc_ci)
    matrix[5][4] = np.mean(W_EI_tc_tr)
    # Sixth column - Structure TR
    matrix[0][5] = np.mean(W_IE_tr_s)
    matrix[1][5] = np.mean(W_IE_tr_m)
    matrix[2][5] = np.mean(W_IE_tr_d)
    matrix[3][5] = np.mean(W_II_tr_ci)
    matrix[4][5] = np.mean(W_IE_tr_tc)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ci': W_II_ci,
        'W_II_tr': W_II_tr,
        'W_EE_tc': W_EE_tc,
        
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_tr': W_EI_s_tr,
        'W_EE_s_tc': W_EE_s_tc,
        
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_tr': W_EI_m_tr,
        'W_EE_m_tc': W_EE_m_tc,
        
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_tr': W_EI_d_tr,
        'W_EE_d_tc': W_EE_d_tc,
        
        'W_IE_ci_s': W_IE_ci_s,
        'W_IE_ci_m': W_IE_ci_m,
        'W_IE_ci_d': W_IE_ci_d,
        'W_II_ci_tr': W_II_ci_tr,
        'W_IE_ci_tc': W_IE_ci_tc,
        
        'W_IE_tr_s': W_IE_tr_s,
        'W_IE_tr_m': W_IE_tr_m,
        'W_IE_tr_d': W_IE_tr_d,
        'W_II_tr_ci': W_II_tr_ci,
        'W_IE_tr_tc': W_IE_tr_tc,
        
        'W_EE_tc_s': W_EE_tc_s,
        'W_EE_tc_m': W_EE_tc_m,
        'W_EE_tc_d': W_EE_tc_d,
        'W_EI_tc_ci': W_EI_tc_ci,
        'W_EI_tc_tr': W_EI_tc_tr,
        }
    
    return { 'matrix': matrix, 'weights': weights }
    
    
def coupling_matrix_PD():    
    neuron_quantities = TCM_model_parameters()['neuron_quantities']
    facilitating_factor = TCM_model_parameters()['connectivity_factor_PD_condition']
    
    n_s = neuron_quantities['S']
    n_m = neuron_quantities['M']
    n_d = neuron_quantities['D']
    n_ci = neuron_quantities['CI']
    n_tc = neuron_quantities['TC']
    n_tr = neuron_quantities['TR']
    
    initial = 0
    final = 1
    interval = final - initial
    
    # =============================================================================
    #     These are to restrict the normalized distribution variance or deviation from the mean
    # =============================================================================
    r_s = initial + interval*np.random.rand(n_s, 1)
    r_m = initial + interval*np.random.rand(n_m, 1)
    r_d = initial + interval*np.random.rand(n_d, 1)
    r_ci = initial + interval*np.random.rand(n_ci, 1)
    r_tr = initial + interval*np.random.rand(n_tr, 1)
    r_tc = initial + interval*np.random.rand(n_tc, 1)
    
    # =============================================================================
    #     COUPLING STRENGTHs within each structure (The same in Normal and PD)
    # EE -> Excitatory to Excitatory  
    # II -> Inhibitory to Inhibitory 
    # =============================================================================
    ## Layer S (was -1e-2 for IEEE paper)
    aee_s = -5e1/facilitating_factor;           W_EE_s = aee_s*r_s;
    ## Layer M (was -1e-2 for IEEE paper)
    aee_m = -5e1/facilitating_factor;           W_EE_m = aee_m*r_m;
    ## Layer D (was -1e-2 for IEEE paper)
    aee_d = -5e1/facilitating_factor;           W_EE_d = aee_d*r_d;
    ## INs 
    aii_ci = -5e1/facilitating_factor;          W_II_ci = aii_ci*r_ci;
    ## Reticular cells
    aii_tr = -5e1/facilitating_factor;          W_II_tr = aii_tr*r_tr;
    ## Relay cells
    aee_tc = 0/facilitating_factor;             W_EE_tc = aee_tc*r_tc;
    
    # =============================================================================
    #     COUPLING STRENGTHs between structures
    # =============================================================================
    #     S
    # =============================================================================
    # M to S coupling
    aee_sm = 3e2/facilitating_factor;          W_EE_s_m = aee_sm*r_s;
    # D to S coupling
    aee_sd = 5e2/facilitating_factor;          W_EE_s_d = aee_sd*r_s;
    # CI (INs) to S coupling
    aei_sci = -7.5e2/facilitating_factor;      W_EI_s_ci = aei_sci*r_s;
    # Reticular to S coupling
    aei_str = 0/facilitating_factor;           W_EI_s_tr = aei_str*r_s;
    # Rel. to S couplings
    aee_stc = 0/facilitating_factor;           W_EE_s_tc = aee_stc*r_s;     
    # =============================================================================
    #     M
    # =============================================================================
    # S to M
    aee_ms = 1e1/facilitating_factor;       W_EE_m_s = aee_ms*r_m; 
    # D to M couplings
    aee_md = 0/facilitating_factor;         W_EE_m_d = aee_md*r_m;            
    # INs to M couplings
    aei_mci = -7.5e2/facilitating_factor;   W_EI_m_ci = aei_mci*r_m;
    # Ret. to M couplings    
    aei_mtr = 0/facilitating_factor;        W_EI_m_tr = aei_mtr*r_m;
    # Rel. to M couplings
    aee_mtc = 0/facilitating_factor;        W_EE_m_tc = aee_mtc*r_m;
    # =============================================================================
    #     D
    # =============================================================================
    # S to D couplings
    aee_ds = 3e2/facilitating_factor;       W_EE_d_s = aee_ds*r_d;
    # M to D couplings
    aee_dm = 0/facilitating_factor;         W_EE_d_m = aee_dm*r_d;
    # INs to D couplings
    aei_dci = -5e3/facilitating_factor;     W_EI_d_ci = aei_dci*r_d;
    # Ret. to D couplings
    aei_dtr = 0/facilitating_factor;        W_EI_d_tr = aei_dtr*r_d;
    # Rel. to D couplings
    aee_dtc = 1e3/facilitating_factor;      W_EE_d_tc = aee_dtc*r_d;
    # =============================================================================
    #     INs (CI)
    # =============================================================================
    # S to INs couplings
    aie_cis = 2e2/facilitating_factor;      W_IE_ci_s = aie_cis*r_ci;
    # M to INs couplings
    aie_cim = 2e2/facilitating_factor;      W_IE_ci_m = aie_cim*r_ci;
    # D to INs couplings
    aie_cid = 2e2/facilitating_factor;      W_IE_ci_d = aie_cid*r_ci;
    # Ret. to INs couplings
    aii_citr = 0/facilitating_factor;       W_II_ci_tr = aii_citr*r_ci;
    # Rel. to INs couplings
    aie_citc = 1e3/facilitating_factor;     W_IE_ci_tc = aie_citc*r_ci;
    # =============================================================================
    #     Reticular
    # =============================================================================
    # S to Ret couplings
    aie_trs = 0/facilitating_factor;        W_IE_tr_s = aie_trs*r_tr;
    # M to Ret couplings
    aie_trm = 0/facilitating_factor;        W_IE_tr_m = aie_trm*r_tr;
    # D to Ret couplings
    aie_trd = 1e2/facilitating_factor;      W_IE_tr_d = aie_trd*r_tr;
    # Ret. Ret INs couplings
    aii_trci = 0/facilitating_factor;       W_II_tr_ci = aii_trci*r_tr;
    # Rel. Ret INs couplings
    aie_trtc = 5e2/facilitating_factor;     W_IE_tr_tc = aie_trtc*r_tr;
    # =============================================================================
    #     Rele
    # =============================================================================
    # S to Rel couplings
    aee_tcs = 0/facilitating_factor;       W_EE_tc_s = aee_tcs*r_tc;   
    # M to Rel couplings
    aee_tcm = 0/facilitating_factor;       W_EE_tc_m = aee_tcm*r_tc;
    # D to Rel couplings
    aee_tcd = 1e2/facilitating_factor;     W_EE_tc_d = aee_tcd*r_tc;
    # INs to Rel couplings
    aei_tcci = 0/facilitating_factor;     W_EI_tc_ci = aei_tcci*r_tc;
    # Ret to Rel couplings
    aei_tctr = -2.5e3/facilitating_factor;  W_EI_tc_tr = aei_tctr*r_tc;
    
    # Initialize matrix (6 structures -> 6x6 matrix)
    matrix = np.zeros((6,6))
    
    # Populating the matrix
    # Main Diagonal
    matrix[0][0] = np.mean(W_EE_s)
    matrix[1][1] = np.mean(W_EE_m)
    matrix[2][2] = np.mean(W_EE_d)
    matrix[3][3] = np.mean(W_II_ci)
    matrix[4][4] = np.mean(W_EE_tc)
    matrix[5][5] = np.mean(W_II_tr)
    # First column - Layer S
    matrix[1][0] = np.mean(W_EE_s_m)
    matrix[2][0] = np.mean(W_EE_s_d)
    matrix[3][0] = np.mean(W_EI_s_ci)
    matrix[4][0] = np.mean(W_EE_s_tc)
    matrix[5][0] = np.mean(W_EI_s_tr)
    # Second column - Layer M
    matrix[0][1] = np.mean(W_EE_m_s)
    matrix[2][1] = np.mean(W_EE_m_d)
    matrix[3][1] = np.mean(W_EI_m_ci)
    matrix[4][1] = np.mean(W_EE_m_tc)
    matrix[5][1] = np.mean(W_EI_m_tr)
    # Thid column - Layer D
    matrix[0][2] = np.mean(W_EE_d_s)
    matrix[1][2] = np.mean(W_EE_d_m)
    matrix[3][2] = np.mean(W_EI_d_ci)
    matrix[4][2] = np.mean(W_EE_d_tc)
    matrix[5][2] = np.mean(W_EI_d_tr)
    # Fourth column - Structure CI
    matrix[0][3] = np.mean(W_IE_ci_s)
    matrix[1][3] = np.mean(W_IE_ci_m)
    matrix[2][3] = np.mean(W_IE_ci_d)
    matrix[4][3] = np.mean(W_IE_ci_tc)
    matrix[5][3] = np.mean(W_II_ci_tr)
    # Fifth column - Structure TCR
    matrix[0][4] = np.mean(W_EE_tc_s)
    matrix[1][4] = np.mean(W_EE_tc_m)
    matrix[2][4] = np.mean(W_EE_tc_d)
    matrix[3][4] = np.mean(W_EI_tc_ci)
    matrix[5][4] = np.mean(W_EI_tc_tr)
    # Sixth column - Structure TRN
    matrix[0][5] = np.mean(W_IE_tr_s)
    matrix[1][5] = np.mean(W_IE_tr_m)
    matrix[2][5] = np.mean(W_IE_tr_d)
    matrix[3][5] = np.mean(W_II_tr_ci)
    matrix[4][5] = np.mean(W_IE_tr_tc)
    
    weights = {
        'W_EE_s': W_EE_s,
        'W_EE_m': W_EE_m,
        'W_EE_d': W_EE_d,
        'W_II_ci': W_II_ci,
        'W_II_tr': W_II_tr,
        'W_EE_tc': W_EE_tc,
        
        'W_EE_s_m': W_EE_s_m,
        'W_EE_s_d': W_EE_s_d,
        'W_EI_s_ci': W_EI_s_ci,
        'W_EI_s_tr': W_EI_s_tr,
        'W_EE_s_tc': W_EE_s_tc,
        
        'W_EE_m_s': W_EE_m_s,
        'W_EE_m_d': W_EE_m_d,
        'W_EI_m_ci': W_EI_m_ci,
        'W_EI_m_tr': W_EI_m_tr,
        'W_EE_m_tc': W_EE_m_tc,
        
        'W_EE_d_s': W_EE_d_s,
        'W_EE_d_m': W_EE_d_m,
        'W_EI_d_ci': W_EI_d_ci,
        'W_EI_d_tr': W_EI_d_tr,
        'W_EE_d_tc': W_EE_d_tc,
        
        'W_IE_ci_s': W_IE_ci_s,
        'W_IE_ci_m': W_IE_ci_m,
        'W_IE_ci_d': W_IE_ci_d,
        'W_II_ci_tr': W_II_ci_tr,
        'W_IE_ci_tc': W_IE_ci_tc,
        
        'W_IE_tr_s': W_IE_tr_s,
        'W_IE_tr_m': W_IE_tr_m,
        'W_IE_tr_d': W_IE_tr_d,
        'W_II_tr_ci': W_II_tr_ci,
        'W_IE_tr_tc': W_IE_tr_tc,
        
        'W_EE_tc_s': W_EE_tc_s,
        'W_EE_tc_m': W_EE_tc_m,
        'W_EE_tc_d': W_EE_tc_d,
        'W_EI_tc_ci': W_EI_tc_ci,
        'W_EI_tc_tr': W_EI_tc_tr,
        }
    
    return { 'matrix': matrix, 'weights': weights }