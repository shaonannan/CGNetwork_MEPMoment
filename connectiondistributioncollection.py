"""
Container that organizes connection components for a simulation, to reduce redundancy

In a simulation, connections that share the same weights and probabilities as well as the same target bin edges,
can make use of the same flux_matrix and threshold_flux_vector. This can significantly improve
the overall memory efficiency of the simulation. To facilitate this,each simulation creates a ConnectionDistributionCollection
object that indexes the ConnectionDistribution objects according to their signature, and re-uses for multiple connections

This version is used for Pytorch and have been test on Dell
"""
class ConnectionDistributionCollection(dict):

    def add_unique_connection(self,cd):
        if not cd.signature in self.keys():
            self[cd.signature] = cd  

"""
we only mark with post(target) population, so the target voltage bin (voltage distribution) is same
for each pre(source) population.

So, if the source(pre) populations hold the coincident  weight,probability as well as
pre-voltage bins, their contribution to the voltaget distribution of pos-population should be
summed up and treated as a ensemble
"""