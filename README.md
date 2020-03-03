# CGNetwork_MEPMoment
SZT Dimensional Reduction of Emergent Spatiotemporal Cortical Dynamics via a Maximum Entropy Moment Closure
# CG Network for V1 via a Maximum Entropy Moment Closure
The code is tested in Anaconda 3 (Python 3.6 and Python 3.7) on Windows 10.

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
