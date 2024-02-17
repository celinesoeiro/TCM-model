"""

@author: Celine Soeiro

@description: Izhikevich neurons parameters used by FL Model

Tonic Spiking:
    a = 0.02 | b = 0.2 | c = -65 | d = 6 | I = [14, 14] | Vp = 30 | V = -70 | tau = 0.25

Phasic Spiking:
    a = 0.02 | b = 0.25 | c = -65 | d = 6 | I = 0.5 | Vp = 30 | V = -64 | tau = 0.25
    
Tonic Bursting:
    a = 0.02 | b = 0.2 | c = -50 | d = 2 | I = 15 | Vp = 30 | V = -70 | tau = 0.25
    
LTS:
    a = 0.02 | b = 0.25 | c = -65 | d = 2 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25
    
TC:
    a = 0.02 | b = 0.25 | c = -65 | d = 0.05 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25

TR: 
    a = 0.02 | b = 0.25 | c = -65 | d = 2.05 | Idc = 0 | Vp = 30 | V = -70 | tau = 0.25
    
"""

from izhikevich_neuron import izhikevich_neuron

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


regular_spiking_neuron, I_RS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 3.5, 
    current_start = 0, 
    current_finish = 10000,
    )

intrinsically_bursting_neuron, I_IB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 4}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 5, 
    current_start = 500, 
    current_finish = 10000,
    )

chattering_neuron, I_CH = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 10, 
    current_start = 500, 
    current_finish = 10000,
    )

fast_spiking_neuron, I_FS = izhikevich_neuron(
    params = {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2}, 
    neuron_type = 'inhibitory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 2, 
    current_start = 0, 
    current_finish = 10000,
    )

thalamo_cortical_depolarized_neuron, I_TC_d = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2.05}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 2, 
    current_start = 0, 
    current_finish = 4000,
    initial_voltage = -60
    )

thalamo_cortical_hyperpolarized_neuron, I_TC_h = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 5, 
    current_start = 500, 
    current_finish = 10000,
    initial_voltage = -90
    )

resonator_neuron, I_RZ = izhikevich_neuron(
    params = {'a': 0.1, 'b': 0.26, 'c': -65, 'd': -1}, 
    neuron_type = 'excitatory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 10, 
    current_start = 500, 
    current_finish = 10000,
    )

low_threshold_spiking_neuron, I_LTS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2}, 
    neuron_type = 'inhibitory', 
    voltage_pick = 30, 
    simulation_time = 100, 
    time_step = 0.01, 
    current_value = 2, 
    current_start = 500, 
    current_finish = 10000,
    )

# =============================================================================

tonic_spiking_neuron, I_TS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
    neuron_type = 'excitatory',
    simulation_time = 10,
    time_step = 0.01,
    current_value = 14,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    initial_voltage = -70,
    )

phasic_spiking_neuron, I_PS = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6},
    neuron_type = 'excitatory',
    current_value = 0.5,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    initial_voltage = -64,
    simulation_time = 10,
    time_step = 0.01,
    )

tonic_bursting_neuron, I_TB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
    current_value = 15,
    current_start = 100,
    current_finish = 900,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.01,
    neuron_type = 'excitatory'
    )

phasic_bursting_neuron, I_PB = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.25, 'c': -55, 'd': 0.05},
    current_value = 0.6,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -64,
    simulation_time = 100,
    time_step = 0.2,
    neuron_type = 'excitatory'
    )

mixed_mode_neuron, I_MM = izhikevich_neuron(
    params = {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
    current_value = 10,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neuron_type = 'excitatory'
    )

spike_frequency_adapt, I_SF = izhikevich_neuron(
    params = {'a': 0.01, 'b': 0.2, 'c': -65, 'd': 8},
    current_value = 30,
    current_start = 50,
    current_finish = 200,
    voltage_pick = 30,
    initial_voltage = -70,
    simulation_time = 100,
    time_step = 0.25,
    neuron_type = 'excitatory'
    )


# Plots
plt.figure(1)
plt.suptitle("Izhikevich model - Regular Spiking")
plt.subplot(2,1,1)
plt.plot(regular_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_RS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(2)
plt.suptitle("Izhikevich model - Intrinically Bursting")
plt.subplot(2,1,1)
plt.plot(intrinsically_bursting_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_IB, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(3)
plt.suptitle("Izhikevich model - Chattering")
plt.subplot(2,1,1)
plt.plot(chattering_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_CH, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(4)
plt.suptitle("Izhikevich model - Fast Spiking")
plt.subplot(2,1,1)
plt.plot(fast_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_FS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(5)
plt.suptitle("Izhikevich model - Thalamo-Cortical Depolarized")
plt.subplot(2,1,1)
plt.plot(thalamo_cortical_depolarized_neuron, 'b', label='voltage (mV)')
plt.yticks([20,0,-20,-40,-60,-80,-100])
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_TC_d, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(6)
plt.suptitle("Izhikevich model - Thalamo-Cortical Hiperpolarized")
plt.subplot(2,1,1)
plt.plot(thalamo_cortical_hyperpolarized_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_TC_h, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(7)
plt.suptitle("Izhikevich model - Resonator")
plt.subplot(2,1,1)
plt.plot(resonator_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_RZ, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(8)
plt.suptitle("Izhikevich model - Low-Threshold Spiking")
plt.subplot(2,1,1)
plt.plot(low_threshold_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(I_LTS, label='Current')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

# =============================================================================
# Advanced Models
# =============================================================================

plt.figure(9)
plt.suptitle("Izhikevich model - Tonic Spiking")
plt.plot(tonic_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(10)
plt.suptitle("Izhikevich model - Phasic Spiking")
plt.plot(phasic_spiking_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(11)
plt.suptitle("Izhikevich model - Tonic Bursting")
plt.plot(tonic_bursting_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(12)
plt.suptitle("Izhikevich model - Phasic Bursting")
plt.plot(phasic_bursting_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(13)
plt.suptitle("Izhikevich model - Mixed Mode")
plt.plot(mixed_mode_neuron, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()

plt.figure(14)
plt.suptitle("Izhikevich model - Spike Frequency Adaptation")
plt.plot(spike_frequency_adapt, 'b', label='voltage (mV)')
plt.legend(shadow=True, fancybox=True)
plt.grid(True)
plt.show()