"""
@author: Celine Soeiro
@description: TCM functions
"""

import numpy as np
from scipy.signal  import butter, lfilter, welch
from math import pi

# =============================================================================
# Izhikevich neuron equations
# =============================================================================
def izhikevich_dvdt(v, u, I):
    return 0.04*v**2 + 5*v + 140 - u + I

def izhikevich_dudt(v, u, a, b):
    return a*(b*v - u)

# =============================================================================
# TM synapse
# =============================================================================
def tm_synapse_eq(u, R, I, AP, t_f, t_d, t_s, U, A, dt, p):
    # Solve EDOs using Euler method
    for j in range(p):
        # u -> utilization factor -> resources ready for use
        u[0][j] = u[0][j - 1] + -dt*u[0][j - 1]/t_f[j] + U[j]*(1 - u[0][j - 1])*AP
        # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
        R[0][j] = R[0][j - 1] + dt*(1 - R[0][j - 1])/t_d[j] - u[0][j]*R[0][j - 1]*AP
        # PSC
        I[0][j] = I[0][j - 1] + -dt*I[0][j - 1]/t_s + A[j]*R[0][j - 1]*u[0][j]*AP
        
    Ipost = np.sum(I)
    
    tm_syn_inst = dict()
    tm_syn_inst['u'] = u
    tm_syn_inst['R'] = R
    tm_syn_inst['I'] = I
    tm_syn_inst['Ipost'] = np.around(Ipost, decimals=6)
        
    return tm_syn_inst

def tm_synapse_poisson_eq(spikes, sim_steps, dt, t_f, t_d, t_s, U, A, time):
    R = np.zeros((3, sim_steps))
    u = np.zeros((3, sim_steps))
    I = np.zeros((3, sim_steps))
    
    for p in range(3):    
        for i in time:
            ap = 0
            if (spikes[0][i - 1] != 0):
                ap = 1
            # u -> utilization factor -> resources ready for use
            u[p][i] = u[p][i - 1] + -dt*u[p][i - 1]/t_f[p] + U[p]*(1 - u[p][i - 1])*ap
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R[p][i] = R[p][i - 1] + dt*(1 - R[p][i - 1])/t_d[p] - u[p][i - 1]*R[p][i - 1]*ap
            # PSC
            I[p][i] = I[p][i - 1] + -dt*I[p][i - 1]/t_s + A[p]*R[p][i - 1]*u[p][i - 1]*ap
            
        
    Ipost = np.sum(I, 0)
        
    return Ipost

# =============================================================================
# DBS
# =============================================================================
def I_DBS(sim_steps, dt, fs, dbs_freq, td_syn, t_f_E, t_d_E, U_E, t_s_E, A_E):    
    step = int(sim_steps/3) # 1 part is zero, 1 part is dbs and another part is back to zero -> pulse-like
    
    I_dbs = np.zeros((2, sim_steps))
    f_dbs = dbs_freq
    
    dbs_duration = step
    dbs_amplitude = 1   # 1mA
    
    T_dbs = np.round(fs/f_dbs)
    dbs_arr = np.arange(0, dbs_duration, T_dbs)
    I_dbs_full = np.zeros((1, dbs_duration))
    
    for i in dbs_arr:
        I_dbs_full[0][int(i)] = dbs_amplitude
    
    I_dbs_pre = 1*np.concatenate((
        np.zeros((1, step)), 
        I_dbs_full, 
        np.zeros((1, step))
        ),axis=1)
    
    R_dbs = np.zeros((3, sim_steps))
    u_dbs = np.ones((3, sim_steps))
    Is_dbs = np.zeros((3, sim_steps))
    
    for p in range(3):
        for i in range(td_syn, sim_steps - 1):
            # u -> utilization factor -> resources ready for use
            u_dbs[p][i] = u_dbs[p][i - 1] + -dt*u_dbs[p][i - 1]/t_f_E[p] + U_E[p]*(1 - u_dbs[p][i - 1])*I_dbs_pre[0][i- td_syn]
            # x -> availabe resources -> Fraction of resources that remain available after neurotransmitter depletion
            R_dbs[p][i] = R_dbs[p][i - 1] + dt*(1 - R_dbs[p][i - 1])/t_d_E[p] - u_dbs[p][i - 1]*R_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
            # PSC
            Is_dbs[p][i] = Is_dbs[p][i - 1] + -dt*Is_dbs[p][i - 1]/t_s_E + A_E[p]*R_dbs[p][i - 1]*u_dbs[p][i - 1]*I_dbs_pre[0][i- td_syn]
            
    I_dbs_post = np.sum(Is_dbs, 0)
    
    I_dbs[0] = I_dbs_pre[0]
    I_dbs[1] = I_dbs_post
    
    return I_dbs

# =============================================================================
# POISSON
# =============================================================================
def homogeneous_poisson(rate, tmax, bin_size): 
    nbins = np.floor(tmax/bin_size).astype(int) 
    prob_of_spike = rate * bin_size 
    spikes = np.random.rand(nbins) < prob_of_spike 
    return spikes * 1

def poisson_spike_generator(num_steps, dt, num_neurons, thalamic_firing_rate, current_value=None):
    # Initialize an array to store spike times for each neuron
    spike_times = [[] for _ in range(num_neurons)]

    # Calculate firing probability
    firing_prob = thalamic_firing_rate * dt  # Calculate firing probability

    # Generate spikes for each neuron using the Poisson distribution
    for t in range(num_steps):
        for neuron_id in range(num_neurons):
            # Generate a random number between 0 and 1
            rand_num = np.random.rand()
            
            # If the random number is less than the firing probability, spike
            if rand_num < firing_prob:
                spike_times[neuron_id].append(t)
            else: 
                spike_times[neuron_id].append(0)
    
    # Creating a vector to be used as current input
    input_current = np.zeros((1, num_steps))
    for sub_spike in spike_times:
        for spike in sub_spike:
            spike_indice = np.array(spike)
            value = np.random.normal(loc=0.25, scale=0.05)
            input_current[0][spike_indice.astype(int)] = value
                
    return spike_times, input_current

# =============================================================================
# RASTER
# =============================================================================
def make_dict(sim_steps, chop_till, n_neurons, fired):
    clean_sim_steps = np.arange(0, sim_steps - chop_till)
    
    new_length = len(clean_sim_steps)*n_neurons
    neuron = np.zeros((new_length, 3))

    n_aux = 0
    t_aux = 0
    for i in range(new_length):
        if (n_aux == n_neurons):
            n_aux = 0
        
        if (t_aux == len(clean_sim_steps)):
            t_aux = 0
            
        neuron[i][0] = n_aux
        neuron[i][1] = t_aux
        neuron[i][2] = fired[n_aux][t_aux]
            
        n_aux += 1
        t_aux +=1
    
    v_dict = {
        "neuron": neuron[:, 0],
        "time": neuron[:, 1],
        "fired": neuron[:, 2],
        }
    
    return v_dict

def export_spike_dict(n_neuron, sim_steps, chop_till, spikes):
    # Creating a dictionary
    clean_sim_steps = np.arange(0, sim_steps - chop_till)
    neuron = {}
    spike_time = []
    for n in range(n_neuron):
        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = []
    
    # Filling the dictionary with the firing time
    for n in range(n_neuron):
        for t in clean_sim_steps:
            if (spikes[n][t] != 0):
                spike_time.append(int(spikes[n][t]))

        neuron_name = f"neuron_{n}"
        neuron[neuron_name] = np.array(spike_time)
    
    return neuron

# =============================================================================
# SIGNAL ANALYSIS
# =============================================================================
def LFP(E_signal, I_signal):
    rho = 0.27
    r = 100e-6
    #### LFP is the sum of the post-synaptic currents
    LFP = (np.subtract(E_signal, 1*I_signal))/(4*pi*r*rho)

    return LFP

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a

#funçao butter_bandpass_filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    
    return y

def PSD(signal, fs):
    (f, S) = welch(signal, fs, nperseg=10*1024)
    
    return f, S