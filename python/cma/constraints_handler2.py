# -*- coding: utf-8 -*-
"""A collection of boundary and (in future) constraints handling classes.
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals
del absolute_import, division, print_function  #, unicode_literals
# __package__ = 'cma'

import numpy as np

from . import fitness_models
from .utilities import utils

def _new_rank_correlation(X, ranks=None):
    """compute correlation between ranks and <x_best-x_worst, x>-ranking"""
    if ranks is None:  # doesn't make sense?
        ranks = list(range(len(X)))
        imin, imax = 0, len(X) - 1
    else:
        imin, imax = np.argmin(ranks), np.argmax(ranks)

    xlin = np.asarray(X[imax]) - X[imin]
    flin = [sum(xlin * x) for x in X]
    print(imin, imax)
    print(flin, ranks)
    print(utils.ranks(flin), utils.ranks(ranks))
    return fitness_models._kendall_tau(flin, ranks)

class Glinear:
    """currently like linear bound constraint function on variable i"""
    def __init__(self, i, sign=1):
        self.i = i
        self.sign = sign
    def __call__(self, x):
        return self.sign * x[self.i]
        
class RankWeight:
    """A weight that adapts depending of observed values.
    
    The weight is designed to be used in a ranksum.
    """
    def __init__(self, increment=None):
        """increment default is 1.
        
        increment around 0.9 gives smallest axis ratio on the "sphere".
        """
        self.weight = 1
        self.increment = increment if increment is not None else 1
    def update(self, g_vals):
        """update weight depending on sign values of `g_vals`.

        Increment by a constant if all values are larger than zero,
        reset to one otherwise.
        """
        assert len(g_vals) > 1  # otherwise we have no basis for adaptation
        if all([g > 0 for g in g_vals]):
            self.weight += self.increment
        else:
            self.weight = 1
        return self.weight
    
class FSumRanks:
    def __init__(self, f_vec):
        """f_vec is a list or tuple of functions or an integer"""
        if not hasattr(f_vec, '__getitem__'):
            assert not callable(f_vec)
            f_vec = [Glinear(i) for i in range(f_vec)]
        self.f_vec = f_vec
        self.weights = [RankWeight() for _ in range(len(self.f_vec))]
        self._xopt = 0

    def distance(self, x):
        return sum((np.asarray(x) - self._xopt)**2)**0.5

    def __call__(self, X):
        ranks = np.zeros(len(X))
        for w_i, f_i in zip(self.weights, self.f_vec):
            fs = [f_i(x) for x in X]
            w_i.update(fs)
            ranks += w_i.weight * (1 + np.asarray(utils.ranks([np.abs(f) for f in fs])))
        return list((1 + ranks / 1e2) * self.distance(X[0])**2)

class GSumRanks:
    """Compute a rank-based penalty from constraints functions
    """
    def __init__(self, g_vec, increment=None):
        """g_vec is a list or tuple of functions"""
        self.g_vec = g_vec
        self.weights = [RankWeight(increment) for _ in range(len(self.g_vec))]
        self._glimit = 0

    def distance(self, G):
        """G-space distance to feasibility of g-values vector G"""
        G0 = np.asarray(G) - self._glimit
        G1 = (G0 > 0) * G0
        return sum(G1)  # TODO-decide: should here be a square?

    @property
    def distances(self):
        """depends on g-values of last call"""
        g_array = np.asarray(self.g_array).T  # now a row per offspring
        return [self.distance(gs) for gs in g_array]

    def g_transform(self, g):
        """maps g in [0, inf] to [0, 1/2]"""
        return np.tanh(g) / 2.  # np.exp(x) / (1 + np.exp(x)) - 0.5

    def __call__(self, X):
        """return weighted ranksum plus tanh(sum g+) / 2"""
        ranks = np.zeros(len(X))
        self.g_array = []
        if not len(self.g_vec):
            return ranks
        for w_i, g_i in zip(self.weights, self.g_vec):
            gs = [g_i(x) for x in X]
            self.g_array += [gs]
            w_i.update(gs)  # this is why we need to loop over g first
            rs = np.asarray(utils.ranks(gs)) + (1 - sum(np.asarray(gs) <= 0))
            rs[rs < 0] = 0
            ranks += w_i.weight * rs
        return list(ranks - min(ranks) + self.g_transform(self.distances))
       
class FGSumRanks:
    """TODO: prevent negative f-values but how?"""
    def __init__(self, f, g_vec, increment=None):
        self.f = f
        self.granking = GSumRanks(g_vec)
        self.weight = 1
        self.increment = increment if increment is not None else 1

    def threshold(self):
        # TODO: rename to constraints_OK
        # TODO: use 2%itle and reconsider the max weight threshold, max should be a percentile?
        return max([w.weight for w in self.granking.weights]) < 2

    def update(self, g_ranks):
        """update weight depending on feasibility ratios"""
        g_array = np.asarray(self.granking.g_array)
        if np.sum(g_array > 0) < g_array.size / 3 and self.threshold():
            self.weight += self.increment
        elif np.sum(g_array > 0) > g_array.size * 2 / 3:
            self.weight -= self.increment
        self.weight = min((max(g_ranks)), self.weight)  # set upper bound
        self.weight = max((1, self.weight))  # lower bound is 1

    def __call__(self, X):
        """return weighted rank plus small offset"""
        fs = [self.f(x) for x in X]
        if not len(self.granking.g_vec):
            return fs
        g_ranks = self.granking(X)
        self.update(g_ranks)
        fg_offset = -max((min(fs), min(self.granking.distances)))
        return [fg_offset + self.weight * f_rank + g_rank for (f_rank, g_rank) in zip(utils.ranks(fs), g_ranks)]
