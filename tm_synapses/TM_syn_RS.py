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

# Izhikevich neuron model
vp = 30     # voltage peak
vr = -65    # voltage threshold

neuron_type = 'Regular Spiking (RS)'

a = 0.02
b = 0.2
c = -65 + 15*random_factor**2
d = 8 - 0.6*random_factor**2

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