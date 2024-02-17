#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:16:50 2024

@author: celinesoeiro
"""
from model_functions import homogeneous_poisson, tm_syn_excit_dep, tm_syn_excit_fac, tm_syn_inib_dep, tm_syn_inib_fac

import numpy as np
from random import seed, random
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

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

neuron_type = "Disparo Rapido (FS)"

a = 0.1 + 0.008*random_factor
b = 0.2 - 0.005*random_factor
c = -65 
d = 2 

W = 1
# =============================================================================
# Poisson spike gen
# =============================================================================
spikes = homogeneous_poisson(rate, tmax, bin_size) 
time = np.arange(1,len(spikes)) * bin_size 
sim_steps = len(spikes)

# =============================================================================
# EXCITATORY - DEPRESSION
# =============================================================================
tm_syn_excit_dep(sim_steps, vr, vp, a, b, c, d, spikes, time, W, dt, neuron_type)

# =============================================================================
# EXCITATORY - FACILITATION
# =============================================================================
tm_syn_excit_fac(sim_steps, vr, vp, a, b, c, d, time, dt, spikes, W, neuron_type  )

# =============================================================================
# INHIBITORY - DEPRESSION
# =============================================================================
tm_syn_inib_dep(sim_steps, dt, time, a, b, c, d, vp, vr, spikes, W, neuron_type )

# =============================================================================
# INHIBITORY - FACILITATION
# =============================================================================
tm_syn_inib_fac(sim_steps, time, dt, a, b, c, d, vp, vr, spikes, W, neuron_type )