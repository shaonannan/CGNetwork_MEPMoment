"""
Module describe dynamics of External Populations

more specifically, External Populations hold the properties like, 
firing rate: eta, weight: w, prob: p, time_bin: dt
To have identical format of Interpopulation and being more convenient to get results
we still give some redundant items, like v1 and nmda!


This version is used for Pytorch and have been test on Dell
"""
import numpy as np

"""
edited by SYX

external population (sub-population)
represent mLGN inputs, real visual stimulus convolve spatiotemporal kernel, thus generating time trials
--> external population is equipped with ...
"""
# External Feedforward Input
class ExternalPopulation(object):
    """
    Parameters:
    etaE/I
    record:
    flag for recoding firing rate or not
    """
    def __init__(self,firing_rate,dt,record=False,**kwargs):
        self.firing_rate_stream = firing_rate
        self.firing_rate = 0.0
        # may adding function to generate spike/firing rate trial use lambdify!!!
        self.type = 'External'
        self.dt = dt
        # additional data/parameters
        self.metadata = kwargs
        self.inmda = 0.0
        self.hnmda = 0.0
        


        # for long-range connections
        self.tau_r = 0.002*1000.0
        self.tau_d = 0.128*1000.0

        self.v1 = 0.0
        self.total_fp_sigv = 0.0
        self.vpeak = 0.0

        self.total_Inmda = 0.0
        self.rhov = 0.0
        
        
        # initialize in simulation
        self.simulation = None
    def initialize(self):
        self.initialize_firing_rate()
    def update(self):
        self.update_firing_rate()

        # print 'Ext NMDA: ',self.curr_Inmda,' HNMDA: ',self.hnmda
        
    def initialize_firing_rate(self):
        # current time --> in platform
        self.curr_t = self.simulation.t
        
        try:
            self.firing_rate = self.firing_rate_stream[np.int(self.curr_t/self.dt)]
        except:
            self.firing_rate = 100.0/1000.0
    def update_firing_rate(self):
        self.curr_t = self.simulation.t
        
        try:
            self.firing_rate = self.firing_rate_stream[np.int(self.curr_t/self.dt)]
        except:
            self.firing_rate = 100.0
        """
        if self.curr_t >0.3:
            temp = 100+50*np.absolute(np.sin(40*self.curr_t))
            self.firing_rate = self.firing_rate_stream * 1.0
        else:
            self.firing_rate = self.firing_rate_stream * 1.0
        """
    # update own hNMDA and iNMDA, which only depends on curr_firing_rate 
    # in another words, a-subpopulation's hNMDA & iNMDA only depend on itself
    @property
    def curr_firing_rate(self):
        curr_firing_rate = self.firing_rate
        return curr_firing_rate
    @property
    def curr_Inmda(self):
        curr_Inmda = self.inmda
        return self.inmda