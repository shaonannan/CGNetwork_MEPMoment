"""
This module contains Connection class, which include connections 
between source-population and target-population

This version is used for Pytorch and have been tested on Dell
"""

import numpy as np
from connectiondistribution import ConnectionDistribution
import collections
import utilities as util

# Recurrent Connection
class Connection(object):
    """
    Parameters:
    pre-population
    post-population
    nsyn-population
    connection weight
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,pre,post,nsyn,nsyn_post,weights,probs,conn_type, v_min,v_max,dv):
        self.pre_population = pre
        self.post_population = post
        self.nsyn = nsyn  # Number of Pre(sender) population
        self.weights = weights
        self.probs = probs
        self.conn_type = conn_type
        ''' self voltage '''
        self.v_min = v_min
        self.v_max = v_max
        self.dv    = dv
        # multiply probability of connection
        ''' here we should calculate idx_kick/idx_vT/idx trhov(target)
        '''
        
        self.idx_kick  = 0
        self.idx_vT    = 0
        self.trhov     = np.zeros(2000)
        self.nsyn_post = nsyn_post
        self.V_sample_post= None
        self.ei_pop_post = 'i'
        
        # get v
        
        
        """
        1) connection_list should be classified into some unique population(cluster)
        which means, if 'weight''syn''prob' is identical,should be classified into identical 
        connection_distribution
        2) curr_firing_rate could be replace by ...
        3) simulation could be used to find original platform
        """
        # initialize None and Initialize when simulation
        self.firing_rate = 0.0
        self.simulation = None
        # long range
        self.inmda = 0.0
        """
        unchanged
        1) flux_matrix and threshold_flux_matrix,
        if connection has identical weight syn and prob, then the clux  matrix
        should be identical, this could be reuse --> connection_distribution
        """
    # initialize by hand! when start simulation
    def initialize(self):
        self.initialize_connection_distribution()
        self.initialize_firing_rate()
        self.initialize_I_nmda()
        self.initialize_MFE()
    
    def initialize_connection_distribution(self):
        CD = ConnectionDistribution(self.post_population.edges,self.weights,self.probs)
        CD.simulation = self.simulation
        self.simulation.connection_distribution_collection.add_unique_connection(CD)
        self.connection_distribution = self.simulation.connection_distribution_collection[CD.signature]
        
        
    def initialize_firing_rate(self):
        self.firing_rate = self.pre_population.curr_firing_rate
    # LONG RANGE 
    def initialize_I_nmda(self):
        self.inmda = self.pre_population.curr_Inmda
    def initialize_MFE(self):
        self.trhov = self.post_population.curr_rhov
        vT = 1.0
        value_kick = vT - self.weights 
        # get v 
        Vedges = util.get_v_edges(self.v_min,self.v_max,self.dv)
        Vbins  = 0.5*(Vedges[0:-1] + Vedges[1:])
        
        idx_vT = len(Vedges)-1 # len(Vbins)

        Ind_k1 = np.where(Vedges>value_kick)
        if np.shape(Ind_k1)[1]>0:
            idx_kick  = Ind_k1[0][0]
        else:
            idx_kick  = idx_vT
        
        self.idx_kick = idx_kick
        self.idx_vT   = idx_vT
        self.V_sample_post = self.post_population.curr_V_sample
        self.ei_pop_post = self.post_population.ei_pop
    def update(self):
        self.firing_rate   =  self.pre_population.curr_firing_rate
        self.inmda         =  self.pre_population.curr_Inmda
        self.trhov         =  self.post_population.curr_rhov
        self.V_sample_post =  self.post_population.curr_V_sample
        # initialize_firing_rate
    def update_connection(self,npre,npost,nsyn,**nkwargs):
        self.pre_population = [],
        self.pre_population = npre,
        self.post_population = [],
        self.post_population = npost,
        self.syn_population = [],
        self.syn_population = nsyn
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    
    @property
    def curr_Inmda(self):
        return self.inmda
    
    @property
    def curr_trhov(self):
        return self.trhov
    
    @property
    def curr_tV_sample(self):
        return self.V_sample_post
    

       
