from model_functions import homogeneous_poisson, tm_syn_excit_dep, tm_syn_excit_fac, tm_syn_inib_dep, tm_syn_inib_fac

import numpy as np
from random import seed, random

seed(1)
random_factor = random()

ms = 1000           # 1ms
rate = 20 * 1/ms    # spike rate 
bin_size = 1        # bin size 
tmax = 1 * ms       # the total lenght of the spike train
dt = rate

neuron_type = "Thalamo-Cortical (TC)"

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold
a = 0.02 + 0.008*random_factor
b = 0.25 - 0.005*random_factor
c = -65 
d = 0.05 

W = 1e2

# =============================================================================
# Poisson spike gen
# =============================================================================
spikes = homogeneous_poisson(rate, tmax, bin_size)
sim_steps = len(spikes) 
time = np.arange(1, sim_steps) * bin_size 

# =============================================================================
# EXCITATORY - DEPRESSION
# =============================================================================
tm_syn_excit_dep(sim_steps, vr, vp, a, b, c, d, spikes, time, W, dt, neuron_type)

# =============================================================================
# EXCITATORY - FACILITATION
# =============================================================================
tm_syn_excit_fac(sim_steps, vr, vp, a, b, c, d, time, dt, spikes, W, neuron_type)

# =============================================================================
# INHIBITORY - DEPRESSION
# =============================================================================
tm_syn_inib_dep(sim_steps, dt, time, a, b, c, d, vp, vr, spikes, W, neuron_type)

# =============================================================================
# INHIBITORY - FACILITATION
# =============================================================================
tm_syn_inib_fac(sim_steps, time, dt, a, b, c, d, vp, vr, spikes, W, neuron_type)
