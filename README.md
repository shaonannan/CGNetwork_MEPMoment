# CGNetwork_MEPMoment
SZT Dimensional Reduction of Emergent Spatiotemporal Cortical Dynamics via a Maximum Entropy Moment Closure
# CG Network for V1 via a Maximum Entropy Moment Closure
The code is tested in Anaconda 3 (Python 3.6 and Python 3.7) on Windows 10.

We also provide the large code (conductance-based I&F spiking neuron model mentioned in our paper), details are in another  repository on this homepage,
https://github.com/shaonannan/largecode_IFSpiking

The structure of this Python program uses Allen's simulation platform DiPDE for reference. 
DiPDE (dipde) is also an excellent simulation platform for numerically solving the time evolution of coupled networks of neuronal populations. You can find their algorithms in,

Website: © 2015 Allen Institute for Brain Science. DiPDE Simulator [Internet]. Available from: https://github.com/AllenInstitute/dipde.

We are still refining this code, especially the 'MFEs' parts (simulation.py, function 'def getMFE_ifdyn' in utilities.py and etc.) which account for transient dynamics. Another project we are following up will be updated on this page.

## Get started with default configurations
In EXAMPLE.py file
    
    cmd = 'python RunVoltageMoment.py -stimulus 0'
Use network with default parameters to run simulation in counterchanging on/off stimulus

    cmd = 'python RunVoltageMoment.py -stimulus 1'
Use network with default parameters to run simulation under ON-/OFF-LMI stimulus

You can also change the parameters by adding other parameters in the command, e.g. changing $S_{EE}$

    cmd = 'python RunVoltageMoment.py -stimulus 1 -see 1.20'

Simulating results in Code_directory/... and save as .mat file.

## Network and Maximum Entropy Moment
See 'RunVoltageMoment.py' for the architecture of our CG network

See 'internalpopulation.py' and 'utilities.py' for our reduced method (Maximum Entropy Moment closure)

See the source code to know details about how this algorithm works.
