# Thalamo Cortical Model with DBS simulation

## What is in it?
- Izhikevich neuron model (izhikevich_neuron folder)
- Tsodkys and Markram synaptic model (tm_synapses folder)
- Farokhniaee and Lowery Thalamo-Cortical model (step-4-* folders)

## How to use?
There are 3 files which contains the model parameters and functions:
- Model parameters (tcm_params file)
- Model functions (model_functions file)
- Model plots (model_plots file)

Use the model parameters file to change the model parameters like simulation overall duration, step, number of neurons and DBS frequency, for example.

This model was accomplished in steps, that's why there are files named step-1-*, step-2-*, step-3-* and step-4*. Below you will find a description of what each step contains:

### Step 1
- TC model with layer D and Cortical Interneurons with thalamic input simulated by an spike generator modeled by Poisson (step-1-cortical).

### Step 2
- Step 1 with layer S neurons (step-2-cortical)

### Step 3
- Step 2 with layer M neurons (step-3-cortical).

### Step 4
- Step 3 with both Thalamo Cortical Nucleus and Thalamic Reticular Nucleus. 
- This is the complete model.
- This step has 2 files, one simulating a normal condition (step-4-thalamus-normal) and another one simulating a parkinsonian condition (step-4-thalamus-PD).



