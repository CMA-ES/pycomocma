#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from hv import HyperVolume
import itertools
import copy

class EmpiricalFront(list):
    """
    A list of objective values in an empirical Pareto front,
    meaning that no point strictly domminates another one in all
    objectives.
    """
    def __init__(self,
                 list_of_f_tuples=None,
                 reference_point=None):
        """
        elements of list_of_f_tuples not in the empirical front are pruned away
        `reference_point` is also used to compute the hypervolume, with the hv
        module of Simon Wessing.

        """
        if list_of_f_tuples is not None and len(list_of_f_tuples):
            try:
                list_of_f_tuples = list_of_f_tuples.tolist()
            except:
                pass
        list.__init__(self, list_of_f_tuples)
        
        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self.prune()  # remove dominated entries, uses self.dominates
        self._set_HV()

    def add(self, f_tuple):
        """add `f_tuple` in `self` if it is not dominated in all objectives.
        """
        f_tuple = list(f_tuple)  # convert array to list
        
        if not self.dominates(f_tuple):
            self += [f_tuple]
        self.prune()
            
    def add_list(self, list_of_f_tuples):
        """
        add list of f_tuples, using the add method.
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
        We assert whether self[idx] dominates f_tuple in all objectives.
        Note that it is independant to the reference_point.
        """
        assert idx < len(self) and idx > -1
        # notion of strong domination here : not the non dominated points,
        # but the points on the empirical front
        if all(self[idx][k] < f_tuple[k] for k in range(len(f_tuple))):
            return True
        return False
    
    def in_domain(self, f_tuple):
        """
        Test if f_tuple dominates the reference point in all objectives.
        """
        if self.reference_point is None:
            raise ValueError("to know the domain, a reference"
                             " point is needed (must be given initially)")
        assert len(f_tuple) == len(self.reference_point)
        if all(f_tuple[k] < self.reference_point[k] for k in range(len(f_tuple))):
            return True
        return False
    
    def dominators(self, f_tuple):
        """return the list of all `f_tuple`-dominating elements in `self`.
        The method 'dominates_with' is used

        >>> from empiricalfront import EmpiricalFront as EF
        >>> a = EF([[1.2, 0.1], [0.5, 1]])
        >>> len(a)
        2
        >>> a.dominators([2, 3]) == a
        True
        >>> a.dominators([0.5, 1])
        [[0.5, 1]]
        >>> len(a.dominators([0.6, 3])), a.dominators([0.6, 3], number_only=True)
        (1, 1)
        >>> a.dominators([0.5, 0.9])
        []
        """ 
        if f_tuple is None:
            return self
        res = []
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                res += [self[idx]]
        return res
        
        
    def kink_points(self, f_tuple = None):
        """
        Create the 'kink' points from elements of self.
        If f_tuple is not None, also add the projections of f_tuple to the empirical front,
        with respect to the axes
        """    
                    
        kinks_loose = []
        dominators_f_tuple = self.dominators(f_tuple)
        for pair in itertools.combinations(dominators_f_tuple, 2):
            kinks_loose += [[max(x) for x in zip(pair[0], pair[1])]]                
        if f_tuple is not None:
            for point in dominators_f_tuple:
                # we project here f_tuple on the axes containing 'point'
                # and collect all the projections
                for idx in range(len(f_tuple)):
                    projected_f_tuple = copy.deepcopy(f_tuple)
                    projected_f_tuple[idx] = point[idx]
                    
                    kinks_loose += [projected_f_tuple]
        kinks = []
        for kink in kinks_loose:
            if not self.dominates(kink):
                kinks += [kink]
        return kinks
    
    
    def prune(self):
        """
        remove point dominated by another one in all objectives.
        """
        for f_tuple in [x for x in self if self.dominates(x)]:
            self.remove(f_tuple)
        if self.reference_point is not None:
            hv_float = HyperVolume(self.reference_point)
            self._hypervolume = hv_float.compute(self)
            
    def _set_HV(self):
        """set current hypervolume value using `self.reference_point`.

        Raise `ValueError` if `self.reference_point` is `None`.

        TODO: we may need to store the list of _contributing_ hypervolumes
        to handle numerical rounding errors later.
        """
        if self.reference_point is None:
            return None
        hv_float = HyperVolume(self.reference_point)
        self._hypervolume = hv_float.compute(self)
        return self._hypervolume
        
    @property
    def hypervolume(self):
        """hypervolume of self with respect to the reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from empiricalfront import EmpiricalFront as EF
        >>> a = EF([[0.5, 0.4], [0.3, 0.7]], [2, 2.1])
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
        Hypervolume improvement of f_tuple with respect to self.
        """
        hv_float = HyperVolume(self.reference_point)
        res1 = hv_float.compute(self + [f_tuple])
        res2 = self._hypervolume
        return res1 - res2
        
    def distance_to_pareto_front(self, f_tuple):
        """
        Compute the distance of a dominated f_tuple to the empirical Pareto front.
        """
        if self.reference_point is None:
            raise ValueError("to compute the distance to the empirical front"
                             "  a reference point is needed (was `None`)")
        if self.in_domain(f_tuple) and not self.dominates(f_tuple):
            return 0  # return minimum distance
        if len(self) == 0:
            return sum([max(0, f_tuple[k] - self.reference_point[k])**2
                        for k in range(len(f_tuple)) ])**0.5
        squared_distances = []
        for kink in self.kinks(f_tuple):
            squared_distances += [sum( (f_tuple[k] - kink[k])**2 for k in range(
                    len(f_tuple)) )]
        return min(squared_distances)**0.5
        
    def hypervolume_improvement(self, f_tuple):
        """return how much `f_tuple` would improve the hypervolume.

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

        


            
    
    
    
    
    