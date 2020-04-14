#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from .hv import HyperVolume
import itertools

import numpy as np

class NonDominatedList(list):
    """
    A list of objective values in an empirical Pareto front,
    meaning that no point strictly domminates another one in all
    objectives.
    
    >>> from nondominatedarchive import NonDominatedList
    >>> a = NonDominatedList([[1,0.9], [0,1], [0,2]], [2, 2])
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
                list_of_f_tuples = [tuple(e) for e in list_of_f_tuples]
            except:
                pass
            list.__init__(self, list_of_f_tuples)
        
        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self.prune()  # remove dominated entries, uses self.dominates
        self._hypervolume = None
        self._kink_points = None

    def add(self, f_tuple):
        """add `f_tuple` in `self` if it is not dominated in all objectives.
        """
        f_tuple = tuple(f_tuple)  # convert array to list
        
        if not self.dominates(f_tuple):
            self.append(f_tuple)
            self._hypervolume = None
            self._kink_points = None
        self.prune()
    
    def remove(self, f_tuple):
        """remove element `f_pair`.
    
        Raises a `ValueError` (like `list`) if ``f_pair is not in self``.
        To avoid the error, checking ``if f_pair is in self`` first is a
        possible coding solution, like
    
        >>> from moarchiving import BiobjectiveNondominatedSortedList
        >>> nda = BiobjectiveNondominatedSortedList([[2, 3]])
        >>> f_pair = [1, 2]
        >>> assert [2, 3] in nda and f_pair not in nda
        >>> if f_pair in nda:
        ...     nda.remove(f_pair)
        >>> nda = BiobjectiveNondominatedSortedList._random_archive(p_ref_point=1)
        >>> for pair in list(nda):
        ...     len_ = len(nda)
        ...     state = nda._state()
        ...     nda.remove(pair)
        ...     assert len(nda) == len_ - 1
        ...     if 100 * pair[0] - int(100 * pair[0]) < 0.7:
        ...         res = nda.add(pair)
        ...         assert all(state[i] == nda._state()[i] for i in [0, 2, 3])
    
        Return `None` (like `list.remove`).
        """
        f_tuple = tuple(f_tuple)  # convert array to list
        list.remove(self, f_tuple)
        self._hypervolume = None
        self._kink_points = None
        
        
    def add_list(self, list_of_f_tuples):
        """
        add list of f_tuples, not using the add method to avoid calling 
        self.prune() several times.
        """
        for f_tuple in list_of_f_tuples:
            f_tuple = tuple(f_tuple)
            if not self.dominates(f_tuple):
                self.append(f_tuple)
                self._hypervolume = None
                self._kink_points = None
        self.prune()        
        
    def prune(self):
        """
        remove point dominated by another one in all objectives.
        """
        for f_tuple in self:
            if not self.in_domain(f_tuple):
                list.remove(self, f_tuple)
        i = 0
        length = len(self)
        while i < length:
            for idx in range(len(self)):
                if self[idx] == self[i]:
                    continue
                if self.dominates_with(idx, self[i]):
                    del self[i]
                    i -= 1
                    length -= 1
                    break
            i += 1
            
            
    def dominates(self, f_tuple):
        """return `True` if any element of `self` dominates or is equal to `f_tuple`.

        Otherwise return `False`.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False
        >>> a._asserts()

        """
        if len(self) == 0:
            return False
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                return True
        return False

    def dominates_with(self, idx, f_tuple):
        """return `True` if ``self[idx]`` dominates or is equal to `f_tuple`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> NDA().dominates_with(0, [1, 2]) is None  # empty NDA
        True
        
        :todo: add more doctests that actually test the functionality and
               not only whether the return value is correct if empty

        """
        if self is None or idx < 0 or idx >= len(self):
            return None
        return self.dominates_with_for(idx, f_tuple)
        
    def dominates_with_old(self, idx, f_tuple):
        ''' deprecated code, now taken over by dominates_wit_for '''
        if all(self[idx][k] <= f_tuple[k] for k in range(len(f_tuple))):
            return True
        return False

    def dominates_with_for(self, idx, f_tuple):
        ''' returns true if self[idx] weakly dominates f_tuple

            replaces dominates_with_old because it turned out
            to run quicker
        '''
        for k in range(len(f_tuple)):
            if self[idx][k] > f_tuple[k]:
                return False
        else:  # yes, indentation is correct, else is not quite necessary in this case
            return True


    def dominators(self, f_tuple, number_only=False):
        """return the list of all `f_tuple`-dominating elements in `self`,

        including an equal element. ``len(....dominators(...))`` is
        hence the number of dominating elements which can also be obtained
        without creating the list with ``number_only=True``.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[1.2, 0.1], [0.5, 1]])
        >>> len(a)
        2
        >>> a.dominators([2, 3]) == a
        True
        >>> a.dominators([0.5, 1])
        [(0.5, 1)]
        >>> len(a.dominators([0.6, 3])), a.dominators([0.6, 3], number_only=True)
        (1, 1)
        >>> a.dominators([0.5, 0.9])
        []

        """
        res = 0 if number_only else []
        for idx in range(len(self)):
            if self.dominates_with(idx, f_tuple):
                if number_only:
                    res += 1
                else:
                    res += [self[idx]]
        return res

    def in_domain(self, f_tuple, reference_point=None):
        """return `True` if `f_tuple` is dominating the reference point,

        `False` otherwise. `True` means that `f_tuple` contributes to
        the hypervolume if not dominated by other elements.

        `f_tuple` may also be an index in `self` in which case
        ``self[f_tuple]`` is tested to be in-domain.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[2.2, 0.1], [0.5, 1]], reference_point=[2, 2])
        >>> assert len(a) == 1
        >>> a.in_domain([0, 0])
        True
        >>> a.in_domain([2, 1])
        False
        >>> all(a.in_domain(ai) for ai in a)
        True
        >>> a.in_domain(0)
        True

        TODO: improve name?
        """
        if reference_point is None:
            reference_point = self.reference_point
        if reference_point is None:
            return True
        try:
            f_tuple = self[f_tuple]
        except TypeError:
            pass
        except IndexError:
            raise  # return None
        if any(f_tuple[k] >= reference_point[k] for k in range(len(reference_point))):
            return False
        return True

    def _strictly_dominates(self, f_tuple):
        """return `True` if any element of `self` strictly dominates `f_tuple`.

        Otherwise return `False`.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False
        >>> a._asserts()

        """
        if len(self) == 0:
            return False
        for idx in range(len(self)):
            if self._strictly_dominates_with(idx, f_tuple):
                return True
        return False

    def _strictly_dominates_with(self, idx, f_tuple):
        """return `True` if ``self[idx]`` strictly dominates `f_tuple`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> NDA()._strictly_dominates_with(0, [1, 2]) is None  # empty NDA
        True

        """
        if idx < 0 or idx >= len(self):
            return None
        if all(self[idx][k] < f_tuple[k] for k in range(len(f_tuple))):
            return True
        return False
        
    def _projection(self, f_tuple, i, x):
        length = len(f_tuple)
        res = length*[0]
        res[i] = x
        for j in range(length):
            if j != i:
                res[j] = f_tuple[j]
        return res
        
    def  _projection_to_empirical_front(self, f_tuple):
        """
        return the orthogonal projections of f_tuple on the empirical front,
        with respect to the coodinates axis. 
        """
        projections_loose = []
        dominators = self.dominators(f_tuple)
        for point in dominators:
            for i in range(len(f_tuple)):
                projections_loose += [self._projection(f_tuple, i, point[i])]
        projections = []
        for proj in projections_loose:
            if not self._strictly_dominates(proj):
                projections += [proj]
        return projections
    
    @property
    def kink_points(self):
        """
        Create the 'kink' points from elements of self.
        If f_tuple is not None, also add the projections of f_tuple
        to the empirical front, with respect to the axes
        """    
        
        if self.reference_point is None:
            raise ValueError("to compute the kink points , a reference"
                        " point is needed (for the extremal kink points)")
        if self._kink_points is not None:
            return self._kink_points
        kinks_loose = []
        for pair in itertools.combinations(self + [self.reference_point], 2):
            kinks_loose += [[max(x) for x in zip(pair[0], pair[1])]]
        kinks = []
        for kink in kinks_loose:
            if not self._strictly_dominates(kink):
                kinks += [kink]
        self._kink_points = kinks
        return self._kink_points
        
    @property
    def hypervolume(self):
        """hypervolume of the entire list w.r.t. the "initial" reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from nondominatedarchive import NonDominatedList as NDA
        >>> a = NDA([[0.5, 0.4], [0.3, 0.7]], [2, 2.1])
        >>> a._asserts()
        >>> a.reference_point == [2, 2.1]
        True
        >>> a._asserts()

        """
        if self.reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (must be given initially)")
        if self._hypervolume is None:
            hv_fraction = HyperVolume(self.reference_point)
            self._hypervolume = hv_fraction.compute(self)
        return self._hypervolume

    def contributing_hypervolume(self, f_tuple):
        """
        Hypervolume improvement of f_tuple with respect to self.
        TODO: the argument should be an index, as in moarchiving.
        """
        if self.reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (must be given initially)")
        hv_fraction = HyperVolume(self.reference_point)
        res1 = hv_fraction.compute(self + [f_tuple])
        res2 = self._hypervolume or hv_fraction.compute(self)
        return res1 - res2
        
    def distance_to_pareto_front(self, f_tuple):
        """
        Compute the distance of a dominated f_tuple to the empirical Pareto front.
        """
        if self.reference_point is None:
            raise ValueError("to compute the distance to the empirical front"
                             "  a reference point is needed (was `None`)")
        if len(self) == 0:
            return sum([max(0, f_tuple[k] - self.reference_point[k])**2
                for k in range(len(f_tuple)) ])**0.5
        if not self.dominates(f_tuple):
            if self.in_domain(f_tuple):
                return 0
            return sum([max(0, f_tuple[k] - self.reference_point[k])**2
                for k in range(len(f_tuple)) ])**0.5
        squared_distances = []
        for kink in self.kink_points:
            squared_distances += [sum( (f_tuple[k] - kink[k])**2 for k in range(
                    len(f_tuple)) )]
        for proj in self._projection_to_empirical_front(f_tuple):
            squared_distances += [sum( (f_tuple[k] - proj[k])**2 for k in range(
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
        return -self.distance_to_pareto_front(f_tuple)

    @staticmethod
    def _random_archive(max_size=500, p_ref_point=0.5):
        from numpy import random as npr
        N = npr.randint(max_size)
        ref_point = list(npr.randn(2) + 1) if npr.rand() < p_ref_point else None
        return NonDominatedList(
            [list(0.01 * npr.randn(2) + npr.rand(1) * [i, -i])
             for i in range(N)],
            reference_point=ref_point)
    
    @staticmethod
    def _random_archive_many(k, max_size=500, p_ref_point=0.5):
        from numpy import random as npr
        N = npr.randint(max_size)
        ref_point = list(npr.randn(k) + 1) if npr.rand() < p_ref_point else None
        return NonDominatedList(
            [list(0.01 * npr.randn(k) + i*(2*npr.rand(k)-1))
             for i in range(N)],
            reference_point=ref_point)

    def _asserts(self):
        """make all kind of consistency assertions.

        >>> import nondominatedarchive
        >>> a = nondominatedarchive.NonDominatedList(
        ...    [[-0.749, -1.188], [-0.557, 1.1076],
        ...    [0.2454, 0.4724], [-1.146, -0.110]], [10, 10])
        >>> a._asserts()
        >>> for p in list(a):
        ...     a.remove(p)
        >>> assert len(a) == 0
        >>> try: a.remove([0, 0])
        ... except ValueError: pass
        ... else: raise AssertionError("remove did not raise ValueError")

        >>> from numpy.random import rand
        >>> for _ in range(120):
        ...     a = nondominatedarchive.NonDominatedList._random_archive()
        ...     if a.reference_point:
        ...         for f_tuple in rand(10, 2):
        ...             h0 = a.hypervolume
        ...             hi = a.hypervolume_improvement(list(f_tuple))
        ...             assert a.hypervolume == h0  # works OK with Fraction

        >>> for _ in range(10):
        ...     for k in range(3,10):                    
        ...         a = nondominatedarchive.NonDominatedList._random_archive_many(k)
        ...         if a.reference_point:
        ...             for f_tuple in rand(10, k):
        ...                 h0 = a.contributing_hypervolume(list(f_tuple))
        ...                 hi = a.hypervolume_improvement(list(f_tuple))
        ...                 assert h0 >= 0            
        ...                 assert h0 == hi or (h0 == 0 and hi < 0)
                            


        """

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Example:
    refpoint = [1.1, 1.1, 1.1]
    myfront = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.25, 0.25, 0.25], [2, 2, 2]]
    emp = NonDominatedList(myfront, refpoint)