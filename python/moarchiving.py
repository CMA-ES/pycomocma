# -*- coding: utf-8 -*-
"""This module contains, for the time being, a single class: a
bi-objective nondominated archive as sorted list with incremental
update in logarithmic time.

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
__license__ = "BSD 3-clause"
__version__ = "0.4.0"
del division, print_function, unicode_literals

import bisect as _bisect # to find the insertion index efficiently
import warnings

class BiobjectiveNondominatedSortedList(list):
    """A sorted list of non-dominated unique objective-pairs.

    Non-domination here means smaller in at least one objective. The list is
    sorted (naturally) by the first objective first.

    The operation

    >>> from moarchiving import BiobjectiveNondominatedSortedList
    >>> any_list = BiobjectiveNondominatedSortedList(any_list)  # doctest:+SKIP

    sorts and prunes the pair list `any_list` to become a consistent
    nondominated sorted archive.

    Afterwards, the methods `add` and `add_list` keep the list always
    in a consistent state. Removing elements with `pop` is consistent too.

    >>> a = BiobjectiveNondominatedSortedList([[1,0.9], [0,1], [0,2]])
    >>> a
    [[0, 1], [1, 0.9]]
    >>> a.add([0, 1])  # doesn't change anything, [0, 1] is not duplicated
    >>> BiobjectiveNondominatedSortedList(
    ...     [[-0.749, -1.188], [-0.557, 1.1076],
    ...     [0.2454, 0.4724], [-1.146, -0.110]])
    [[-1.146, -0.11], [-0.749, -1.188]]
    >>> a._asserts()  # consistency assertions

    TODO: write more example doctest (e.g. with given `reference_point`)

    Details: This list doesn't prevent the user to insert a new element
    anywhere and hence get into an inconsistent state. Inheriting from
    `sortedcontainers.SortedList` would ensure that the `list` remains
    at least sorted.

    See also:
        https://pypi.org/project/sortedcontainers
        https://code.activestate.com/recipes/577197-sortedcollection/
        https://pythontips.com/2016/04/24/python-sorted-collections/

    """
    def __init__(self, list_of_f_pairs=None, reference_point=None, sort=sorted):
        """`list_of_f_pairs` does not need to be sorted.

        f-pairs beyond the `reference_point` are pruned away. The
        `reference_point` is also used to compute the hypervolume.

        ``sort=lambda x: x`` will prevent a sort, which
        can be useful if the list is already sorted.
        """
        self._make_expensive_asserts = True
        if list_of_f_pairs is not None and len(list_of_f_pairs):
            try:
                list_of_f_pairs = list_of_f_pairs.tolist()
            except:
                pass
            if len(list_of_f_pairs[0]) != 2:
                raise ValueError("need elements of len 2, got %s"
                                 " as first element" % str(list_of_f_pairs[0]))
            list.__init__(self, sort(list_of_f_pairs))
        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self.prune()  # remove dominated entries, uses in_domain, hence ref-point
        self._set_hypervolume()
        self._make_expensive_asserts and self._asserts()
        # TODO: here we may want to set up a contributing _hypervolumes_list
        # see compute_hypervolumes

    def add(self, f_pair):
        """insert `f_pair` in `self` if and only if it is nondominated.

        Return index at which the insertion took place or `None`. The
        list remains to be sorted in the process.

        Details: For performance reasons, `insert` is avoided in favor
        of `__setitem__`, if possible.
        """
        f_pair = list(f_pair)  # convert array to list
        if not self.in_domain(f_pair):
            return None
        idx = self.bisect_left(f_pair)
        if self.dominates_with(idx - 1, f_pair) or self.dominates_with(idx, f_pair):
            return None
        assert idx == len(self) or not f_pair == self[idx]
        # here f_pair now is non-dominated
        return self._add_at(idx, f_pair)

    def _add_at(self, idx, f_pair):
        """add `f_pair` at position `idx` iff it is nondominated by its neighbours
        """
        if idx == len(self) or f_pair[1] > self[idx][1]:
            self.insert(idx, f_pair)
            self._add_HV(idx)
            return idx
        # here f_pair now dominates self[idx]
        idx2 = idx + 1
        while idx2 < len(self) and f_pair[1] <= self[idx2][1]:
            # f_pair also dominates self[idx2]
            # self.pop(idx)  # slow
            # del self[idx]  # slow
            idx2 += 1  # delete later in a chunk
        self._subtract_HV(idx, idx2)
        self[idx] = f_pair  # on long lists [.] is much cheaper than insert
        del self[idx + 1:idx2]  # can make `add` 20x faster
        self._add_HV(idx)
        return idx

    def add_list(self, list_of_f_pairs):
        """insert a list of f-pairs which doesn't need to be sorted.

        This is just a shortcut for looping over `add`.

        >>> from moarchiving import BiobjectiveNondominatedSortedList
        >>> arch = BiobjectiveNondominatedSortedList()
        >>> list_of_f_pairs = [[1, 2], [0, 3]]
        >>> for f_pair in list_of_f_pairs:
        ...     arch.add(f_pair)  # return insert index or None
        0
        0
        >>> arch == sorted(list_of_f_pairs)  # both entries are nondominated
        True
        >>> arch.compute_hypervolume([3, 4]) == 5.0
        True

        Return number of actually inserted f-pairs.

        Details: when `list_of_pairs` is already sorted, `merge` may have
        a small performance benefit.
        """
        nb = len(self)
        for f_pair in list_of_f_pairs:
            self.add(f_pair)
        return len(self) - nb

    def merge(self, list_of_f_pairs):
        """merge a sorted list of f-pairs which doesn't need to be nondominated.

        Return number of actually inserted f-pairs.
        """
        raise NotImplementedError("TODO: proof read and test, use `add_list` for the time being")
        nb = len(self)
        idx = 0
        for f_pair in list_of_f_pairs:
            if not self.in_domain(f_pair):
                continue
            f_pair = list(f_pair)  # convert array to list
            idx = self.bisect_left(f_pair, idx)
            self._add_at(idx, f_pair)
        self._set_hypervolume()
        return len(self) - nb

    def bisect_left(self, f_pair, lowest_index=0):
        """return index where `f_pair` may need to be inserted.

        Smaller indices have a strictly better f1 value or equal f1 and better
        f2 value.

        Details: This method does a binary search in `self` using
        `bisect.bisect_left`.
        """
        return _bisect.bisect_left(self, f_pair, lowest_index)

    def dominates(self, f_pair):
        """return `True` if any element of `self` dominates or is equal to `f_pair`.

        Otherwise return `False`.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False

        See also `bisect_left` to find the closest index.
        """
        idx = self.bisect_left(f_pair)
        if self.dominates_with(idx - 1, f_pair) or self.dominates_with(idx, f_pair):
            return True
        return False

    def dominates_with(self, idx, f_pair):
        """return `True` if ``self[idx]`` dominates (or is equal to) `f_pair`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> NDA().dominates_with(0, [1, 2]) is None  # empty NDA
        True

        """
        if idx < 0 or idx >= len(self):
            return None
        if self[idx][0] <= f_pair[0] and self[idx][1] <= f_pair[1]:
            return True
        return False

    def dominators(self, f_pair, number_only=False):
        """return the list of all `f_pair`-dominating elements in `self`,

        including an equal element. ``len(...dominators(...))`` is
        hence the number of dominating elements which can also be obtained
        without creating the list with ``number_only=True``.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[1.2, 0.1], [0.5, 1]])
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
        idx = self.bisect_left(f_pair)
        if idx < len(self) and self[idx] == f_pair:
            res = 1 if number_only else [self[idx]]
        else:
            res = 0 if number_only else []
        idx -= 1
        while idx >= 0 and self[idx][1] <= f_pair[1]:
            if number_only:
                res += 1
            else:
                res.insert(0, self[idx])  # keep sorted
            idx -= 1
        return res

    def in_domain(self, f_pair, reference_point=None):
        """return `True` if `f_pair` is dominating the reference point,

        `False` otherwise. `True` means that `f_pair` contributes to
        the hypervolume.

        TODO: improve name?
        """
        if reference_point is None:
            reference_point = self.reference_point
        if reference_point is None:
            return True
        if (f_pair[0] >= reference_point[0] or
            f_pair[1] >= reference_point[1]):
            return False
        return True

    @property
    def hypervolume(self):
        """return hypervolume w.r.t. the "initial" reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
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
        if self._make_expensive_asserts:
            assert abs(self._hypervolume - self.compute_hypervolume(self.reference_point)) < 1e-12
        return self._hypervolume

    def _set_hypervolume(self):
        """set current hypervolume value using `self.reference_point`.

        Raise `ValueError` if `self.reference_point` is `None`.

        TODO: we may need to store the list of _contributing_ hypervolumes
        to handle numerical rounding errors later.
        """
        if self.reference_point is None:
            return None
        if hasattr(self, '_hypervolumes_list'):
            raise NotImplementedError("update list of hypervolumes")
        self._hypervolume = self.compute_hypervolume(self.reference_point)
        return self._hypervolume

    def compute_hypervolume(self, reference_point):
        """return hypervolume w.r.t. reference_point"""
        if reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (was `None`)")
        hv = 0.0
        idx = 0
        while idx < len(self) and not self.in_domain(self[idx], reference_point):
            idx += 1
        if idx < len(self):
            hv += (reference_point[0] - self[idx][0]) * (reference_point[1] - self[idx][1])
            idx += 1
        while idx < len(self) and self.in_domain(self[idx], reference_point):
            hv += (reference_point[0] - self[idx][0]) * (self[idx - 1][1] - self[idx][1])
            idx += 1
        return hv

    def compute_hypervolumes(self, reference_point):
        """return list of contributing hypervolumes w.r.t. reference_point
        """
        raise NotImplementedError()
        # construct self._hypervolumes_list
        # keep sum of different size elements separate,
        # say, a dict of index lists as indices[1e12] indices[1e6], indices[1], indices[1e-6]...
        hv = {}
        for key in indices:
            hv[key] = sum(_hypervolumes_list[i] for i in indices[key])
        # we may use decimal.Decimal to compute the sum of hv
        decimal.getcontext().prec = 88
        hv_sum = sum([decimal.Decimal(hv[key]) for key in hv])

    def _subtract_HV(self, idx0, idx1):
        """remove contributing hypervolumes of elements self[idx0] to self[idx1 - 1]
        """
        if self.reference_point is None:
            return None
        if idx0 == 0:
            y = self.reference_point[1]
        else:
            y = self[idx0 - 1][1]
        dHV = 0.0
        for idx in range(idx0, idx1):
            if idx == len(self) - 1:
                assert idx < len(self)
                x = self.reference_point[0]
            else:
                x = self[idx + 1][0]
            dHV -= (x - self[idx][0]) * (y - self[idx][1])
        assert dHV <= 0  # and without loss of precision strictly smaller
        if self._hypervolume and abs(dHV) / self._hypervolume < 1e-9:
            warnings.warn("_subtract_HV: %f + %f loses many digits of precision"
                          % (dHV, self._hypervolume))
        self._hypervolume += dHV
        assert self._hypervolume >= 0
        if hasattr(self, '_hypervolumes_list'):
            raise NotImplementedError("update list of hypervolumes")
        return dHV

    def _add_HV(self, idx):
        """add contributing hypervolume of self[idx]"""
        if self.reference_point is None:
            return None
        assert 0 <= idx < len(self)  # idx is not a public interface, hence assert is fine
        if idx == 0:
            y = self.reference_point[1]
        else:
            y = self[idx - 1][1]
        if idx == len(self) - 1:
            x = self.reference_point[0]
        else:
            x = self[idx + 1][0]
        dHV = (x - self[idx][0]) * (y - self[idx][1])
        assert dHV >= 0
        if self._hypervolume and dHV / self._hypervolume < 1e-9:
            warnings.warn("_subtract_HV: %f + %f loses many digits of precision"
                          % (dHV, self._hypervolume))
        self._hypervolume += dHV
        if hasattr(self, '_hypervolumes_list'):
            raise NotImplementedError("update list of hypervolumes")
        return dHV

    def prune(self):
        """remove dominated entries assuming that the list is sorted.

        Return number of dropped elements.
        """
        nb = len(self)
        i = 1
        while i < len(self):
            i0 = i
            while i < len(self) and (self[i][1] >= self[i0 - 1][1] or
                                         not self.in_domain(self[i])):
                i += 1
                # self.pop(i + 1)  # about 10x slower in notebook test
            del self[i0:i]
            i = i0 + 1
        return nb - len(self)

    def _asserts(self):
        """make all kind of consistency assertions"""
        assert sorted(self) == self
        for pair in self:
            assert self.count(pair) == 1
        assert BiobjectiveNondominatedSortedList(self) == self
        for pair in self:
            assert self.dominates(pair)
            assert not self.dominates([v - 0.001 for v in pair])
        if self.reference_point is not None:
            assert abs(self.hypervolume - self.compute_hypervolume(self.reference_point)) < 1e-11


if __name__ == "__main__":
    import doctest
    doctest.testmod()
