"""
Module contains ConnectionDistribution class, clustering connections to identical target cells
in the same connectiondistribution(bin)

This version is used for Pytorch and have been test on Dell
"""

class ConnectionDistribution(object):
    """
    Parameters:
    which could define unique connection,
    like weight, nsyn and prob
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,edges,weights,probs,sparse = True):
        # all properties passed in are torch.Tensor/Variable
        self.edges   = edges   
        self.weights = weights
        self.probs   = probs

        # reversal potential could be used in conductance based model
        self.reversal_potential = None
        if self.reversal_potential != None:
            assert NotImplementedError 
    def initialize(self):
        self.t = 0.0

    @property    
    def signature(self):
        """
        unique signature
        """
        return (tuple(self.edges),tuple([self.weights]),tuple([self.probs]))     