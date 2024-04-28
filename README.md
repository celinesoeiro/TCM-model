# Thalamo Cortical Model with DBS simulation

![MTC_DBS_en](https://github.com/celinesoeiro/model-TC/assets/52112166/7d24394c-82b5-44b2-a17d-419ac2b28409)

## What is in it?
- Izhikevich neuron model (izhikevich_neuron folder)
- Tsodkys and Markram synaptic model (tm_synapses folder)
- Farokhniaee and Lowery Thalamo-Cortical model (step-4-* folders)

## Installation instructions
- First step is to clone the repository using your prefered method
- After that you can choose to run the code using conda: In the requirements-conda.txt file you can find the necessary packages to create an environment to run the project
- **OR** you can use the following command to install the necessary packages using pip command
    
    ``` python3 -m pip install numpy spicy matplotlib seaborn```

## How to use?
There are 3 files which contains the model parameters and functions:
- Model parameters (tcm_params.py file)
- Model functions (model_functions.py file)
- Model plots (model_plots.py file)

Use the model parameters file to change the model parameters like simulation overall duration, step, number of neurons and DBS frequency, for example.

This model was accomplished in steps, that's why there are files named step-1-*, step-2-*, step-3-* and step-4-*. 

To run the complete model you can run one of the following files:
- step-4-tc-normal.py -> To simulate the TCM in a normal condition
- step-4-tc-PD.py -> To simulate the TCM in a Parkinsonian condition
- DBS.py -> To simulate the DBS treatment in a Parkinsonian condition

If you want to follow the step by step approach, below you will find a description of what each step contains:

### Step 1
- TC model with layer D and Cortical Interneurons with thalamic input simulated by an spike generator modeled by Poisson (step-1-cortical).

### Step 2
- Step 1 with layer S neurons (step-2-cortical)

### Step 3
- Step 2 with layer M neurons (step-3-cortical).

### Step 4
- Step 3 with both Thalamo Cortical Nucleus and Thalamic Reticular Nucleus. 
- This is the complete model.
- This step has 2 files, one simulating a normal condition (step-4-tc-normal.py) and another one simulating a parkinsonian condition (step-4-tc-PD.py).

### DBS
- Go to tcm_params file and switch the dbs_freq variable to the desired DBS frequency
- Your simulation time should be a multiple of 3 (because the stimulus is applied in 1/3 of the total time)
- Save changes in the tcm_params file and
- Run the DBS.py file

## Signal analysis
- To obtain the PSD comparison between normal, PD and DBS states you should run the file signal_analysis.py
- To obtain the PSD comparison between the DBS states you should run the file dbs_compare.py

