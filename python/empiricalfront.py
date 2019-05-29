#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from hv import HyperVolume
import itertools

class EmpiricalFront(list):
    """
    """
    def __init__(self,
                 list_of_f_tuples=None,
                 reference_point=None):
        """
        """
        list.__init__(self)
        
        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self.prune()  # remove dominated entries, uses self.dominates
        self._set_HV()

    def add(self, f_tuple):
        """add `f_tuple` in `self` if it is not (weakly) dominated.
        """
        f_tuple = list(f_tuple)  # convert array to list
        
        if not self.dominates(f_tuple):
            self += [f_tuple]
        self.prune()
            
    def add_list(self, list_of_f_tuples):
        """
        """
        for f_tuple in list_of_f_tuples:
            self.add(f_tuple)
    
    def dominates(self, f_tuple):
        """
        """
        if not self.in_domain(f_tuple):
            return True
        
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                return True
        return False

    def dominates_with(self, idx, f_tuple):
        """
        """
        assert idx < len(self)
        if all(self[idx][k] <= f_tuple[k] for k in range(len(self))) and any(
                self[idx][k] < f_tuple[k] for k in range(len(self))):
            return True
        return False
    
    def in_domain(self, f_tuple):
        """
        """
        # raise if refpoint not defined
        assert len(f_tuple) == len(self.reference_pooint)
        if all(f_tuple[k] < self.reference_point[k] for k in range(len(f_tuple))):
            return True
        return False
    
    def kink_points(self, f_tuple = None):
        """
        If f_tuple, also add the orthogonal projections of f_tuple to the 
        empirical front
        """
        kinks_loose = []
        for pair in itertools.combinations(self, 2):
            kinks_loose += [[max(x) for x in zip(pair[0], pair[1])]]
        kinks = []
        for kink in kinks_loose:
            if not self.dominates(kink):
                kinks += [kink]
        return kinks
    
    def 
    
    def prune(self):
        """
        """
        for f_tuple in filter(lambda x: self.dominates(x), self):
            self.remove(f_tuple)
        if self.reference_point is not None:
            hv_float = HyperVolume(self.reference_point)
            self._hypervolume = hv_float(self)
            
    def _set_HV(self):
        """set current hypervolume value using `self.reference_point`.

        Raise `ValueError` if `self.reference_point` is `None`.

        TODO: we may need to store the list of _contributing_ hypervolumes
        to handle numerical rounding errors later.
        """
        if self.reference_point is None:
            return None
        hv_float = HyperVolume(self.reference_point)
        self._hypervolume = hv_float(self)
        return self._hypervolume
        
    @property
    def hypervolume(self):
        """hypervolume of self with respect to the reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from moarchiving import NondominatedSortedList as NDA
        >>> a = NDA([[0.5, 0.4], [0.3, 0.7]], [2, 2.1])
        >>> a._asserts()
        >>> a.reference_point == [2, 2.1]
        True
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True
        >>> a.add([0.2, 0.8])
        0
        >>> a._asserts()
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True
        >>> a.add([0.3, 0.6])
        1
        >>> a._asserts()
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True

        """
        if self.reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (must be given initially)")

        return self._hypervolume
                
                
    def contributing_hypervolume(self, f_tuple):
        """
        """
        hv_float = HyperVolume(self.reference_point)
        res1 = hv_float(self + [f_tuple])
        res2 = self._hypervolume
        return res1 - res2
    
    def distance_to_hypervolume_area(self, f_tuple):
        return (sum(max((0, f_tuple[k] - self.reference_point[k]))**2
                for k in range(self.dim)))**0.5 \
               if self.reference_point else 0
        
    def distance_to_pareto_front(self, f_tuple):
        """
        """
        if self.reference_point is None:
            raise ValueError("to compute the distance to the empirical front"
                             "  a reference point is needed (was `None`)")
        if self.in_domain(f_tuple) and not self.dominates(f_tuple):
            return 0  # return minimum distance
        if len(self) == 0:
            return sum([max(0, f_tuple[k] - self.reference_point[k])**2
                        for k in range(len(f_tuple)) ])**0.5

        raise NotImplementedError()

        
    def hypervolume_improvement(self, f_tuple):
        """return how much `f_tuple` would improve the hypervolumen.

        If dominated, return the distance to the empirical pareto front
        multiplied by -1.
        Else if not in domain, return distance to the reference point
        dominating area times -1.
        """
        contribution = self.contributing_hypervolume(f_tuple)
        assert contribution >= 0
        if contribution:
            return contribution
        return self.distance_to_pareto_front(f_tuple)

        


            
    
    
    
    
    