#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cma
import numpy as np

class RankPenalizedFitness:
    """compute f-values of infeasible solutions as rank_f-inverse(const + sum g-ranks).
    
    The inverse is computed by linear interpolation.
    
    Draw backs: does not support approaching the optimum from the infeasible domain.
    
    Infeasible solutions with valid f-value measurement could get a 1/2-scaled credit for their
    f-rank difference to the base f-value.
    """

    def __init__(self, f_values, g_list_values):
        self.f_values = f_values
        self.g_list_values = g_list_values
        # control parameters
        self.base_prctile = 0.2  # best f-value an infeasible solution can get
        self.g_scale = 1.01  # factor for g-ranks penalty
        self._debugging = False
        # internal state
        self.f_current_best = 0

    def __call__(self):
        """
        Assumes that at least one solution does not return nan as f-value
        """
        # TODO: what to do if there is no f-value for a feasible solution
     #   f_values = [self.f(x) for x in X]
        f_values = self.f_values
        g_ranks_list = []
        is_feasible = np.ones(len(f_values))
   #     for g in self.g_list:
        for g_values in self.g_list_values:
    #        g_values = [g(x) for x in X]         
            g_is_feas = np.asarray(g_values) <= 0
            is_feasible *= g_is_feas
            nb_feas = sum(g_is_feas)
            g_ranks = [g - nb_feas + 1 if g >= nb_feas else 0
                       for g in cma.utilities.utils.ranks(g_values)]  # TODO: this fails with nan-values
            if self._debugging: print(g_ranks)
            g_ranks_list.append(g_ranks)
        idx_feas = np.where(is_feasible)[0]
        # we could also add the distance to the best feasible solution as penalty on the median?
        # or we need to increase the individual g-weight with the number of iterations that no single
        #    feasible value was seen
        # change f-values of infeasible solutions
        sorted_feas_f_values = sorted(np.asarray(f_values)[idx_feas])
        try: self.f_current_best = sorted_feas_f_values[0]
        except IndexError: pass
        j0 = self.base_prctile * (len(idx_feas) - 1)
        #         for i in set(range(len(X))).difference(idx_feas):
        for i in set(range(len(f_values))).difference(idx_feas):
            j = j0 + self.g_scale * (
                    sum(g_ranks[i] for g_ranks in g_ranks_list) - 1)  # -1 makes base a possible value
            assert j >= self.base_prctile * (len(idx_feas) - 1)
            # TODO: use f-value of infeasible solution if available?
            if 11 < 3 and np.isfinite(f_values[i]):
                self.gf_scale = 1 / 2
                j += self.gf_scale * (_interpolated_rank(f_values, f_values[i]) - 
                                      _interpolated_rank(f_values, f_values[j0]))  # TODO: filter f-values by np.isfinite
            j = max((j, 0))
            j1, j2 = int(j), int(np.ceil(j))
            f1 = self._f_from_index(sorted_feas_f_values, j1)
            f2 = self._f_from_index(sorted_feas_f_values, j2)
            # take weighted average fitness between index j and j+1
            f_values[i] = 0e-6 + (j - j1) * f2 + (j2 - j) * f1 if j2 > j1 else f1
        return f_values

    def _f_from_index(self, f_values, i):
        """`i` must be an integer but may be ``>= len(f_values)``"""
        imax = len(f_values) - 1
        if imax < 0:  # no feasible f-value
            return self.f_current_best + i
        return f_values[min((imax, i))] + max((i - imax, 0))
        
