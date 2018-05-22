# -*- coding: utf-8 -*-
"""This module contains, for the time being, a single class: a
bi-objective nondominated archive as sorted list with incremental
update in logarithmic time.

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
__license__ = "BSD 3-clause"
__version__ = "0.3.0"
del division, print_function, unicode_literals

import bisect as _bisect # to find the insertion index efficiently

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

    TODO: write more example doctest

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

        ``sort=lambda x: x`` will prevent a sort, which
        can be useful if the list is already sorted.
        """
        self.reference_point = reference_point
        if list_of_f_pairs is not None and len(list_of_f_pairs):
            try:
                list_of_f_pairs = list_of_f_pairs.tolist()
            except:
                pass
            if len(list_of_f_pairs[0]) != 2:
                raise ValueError("need elements of len 2, got %s"
                                 " as first element" % str(list_of_f_pairs[0]))
            list.__init__(self, sort(list_of_f_pairs))
            self.prune()  # remove dominated entries

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
        if not self:
            self.append(f_pair)
            # TODO: update HV
            return 0
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
            # TODO: add HV
            return idx
        # here f_pair now dominates self[idx]
        idx2 = idx + 1
        while idx2 < len(self) and f_pair[1] <= self[idx2][1]:
            # f_pair also dominates self[idx2]
            # self.pop(idx)  # slow
            # del self[idx]  # slow
            idx2 += 1  # delete later in a chunk
        # TODO: remove HV(idx, idx2)
        del self[idx + 1:idx2]  # can make `add` 20x faster
        self[idx] = f_pair  # on long lists [.] is much cheaper than insert
        # TODO: add HV(idx)
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
        raise NotImplementedError("TODO: proof read and test")
        nb = len(self)
        idx = 0
        for f_pair in list_of_f_pairs:
            if not self.in_domain(f_pair):
                continue
            f_pair = list(f_pair)  # convert array to list
            idx = self.bisect_left(f_pair, idx)
            self._add_at(idx, f_pair)
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

    def in_domain(self, f_pair):
        """TODO: improve name?
        return `True` if `f_pair` is "below" the reference point, `False` otherwise.

        This means `f_pair` contributes to the hypervolume.
        """
        if self.reference_point is None:
            return True
        if (f_pair[0] >= self.reference_point[0] or
            f_pair[1] >= self.reference_point[1]):
            return False
        return True

    @property
    def hypervolume(self):
        """return hypervolume w.r.t. the "initial" reference point.

        Raise `ValueError` when no reference point was given initially.
        """
        try:
            return self._hypervolume
        except AttributeError:
            self._set_hypervolume()
            return self._hypervolume

    def _set_hypervolume(self):
        """set current hypervolume value using `self.reference_point`.

        Raise `ValueError` if `self.reference_point` is `None`.
        """
        self._hypervolume = self._compute_hypervolume(self.reference_point)

    def compute_hypervolume(self, reference_point):
        """return hypervolume w.r.t. reference_point"""
        if reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (was `None`)")
        raise NotImplementedError()
s
    def _subtract_HV(self, idx0, idx1):
        """remove contributing hypervolumes of elements self[idx0] to self[idx1 - 1]
        """
        if self.reference_point is None:
            return
        raise NotImplementedError()

    def _add_HV(self, idx):
        """add contributing hypervolume of self[idx]"""
        if self.reference_point is None:
            return
        raise NotImplementedError()

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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
