#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:17:35 2019

"""

#        
#    def _modulo(self, number_asks):
#        """
#        `number_asks` is an int.
#        
#        returns the list `[self._start_ask % self.num_kernels, ..., 
#        (self._start_ask + number_asks - 1) % self.num_kernels]`.
#        
#        Designed to be used in the `ask` method:
#        self._modulo(number_asks) returns `number asks` successive integers, 
#        starting from `self._start_ask`, and whenever `self.num_kernels` is 
#        reached, we replace it by `0` and continue the sequence from `0`: 
#        it's a torus.
#        
#        Example:
#        --------    
#        If self.num_kernels = 5 and self._start_ask = 3:
#        self._modulo(5) = [3, 4, 0, 1, 2]
#        """
#        res = []
#        assert number_asks > 0
#        for k in range(number_asks):
#            res += [(self._start_ask + k) % self.num_kernels]
#        return res
       
#def order_generator(seq):
#    """the generator for `randint_derandomized`
#    code from the module cocopp, 
#    in: cocopp.toolsstats._randint_derandomized_generator
#    """
#    size = len(seq)
#    delivered = 0
#    while delivered < size:
#        for i in seq:
#            delivered += 1
#            yield i
#            if delivered >= size:
#                break
#            
#class Sequence(object):
#    """
#    TODO: docstring + comments to be done.
#    """
#    def __init__(self, permutation, seq):
#        self.delivered = 0
#        self.permutation = permutation
#        self.seq = seq
#        self.generator = order_generator(permutation(seq))
#    def __call__(self):
#        while True:
#            for i in self.generator:
#                self.delivered += 1
#                yield i
#                if self.delivered % len(self.seq) == 0:
#                    self = Sequence(self.permutation, self.seq)

#class Order(object):
#    """
#    `Order(optimizer)` is a function that takes an index `i` as argument, and returns
#    the opposite contributing hypervolume of `optimizer.kernels[i].objective_values`
#    into `optimizer.pareto_front`.
#    Example:
#        list_of_solvers = mo.get_cmas(num_kernels * [dimension * [0.3]], 0.2)
#        moes = mo.Sofomore(list_of_solvers, reference_point = [11,11])
#        moes.order = Order(moes)
#
#    """
#    def __init__(self, optimizer):
#        self.optimizer = optimizer
#    def __call__(self,i):
#        if self.optimizer.kernels[i].objective_values not in self.optimizer.pareto_front:
#            # meaning that the point is not nondominated
#            return 0
#        else: # the point is nondominated: the (opposite) contributing hypervolume is a non zero value
#            index = self.optimizer.pareto_front.bisect_left(self.optimizer.kernels[i].objective_values)
#            return - self.optimizer.pareto_front.contributing_hypervolume(index)

        #self._order = Sequence(self.options['update_order'], seq)() # generator


#
#    @property
#    def ratio_inactive(self):
#        """
#        return the ratio of inactive kernels among all kernels.
#        """
#        ratio = 0
#        for kernel in self.kernels:
#            if kernel.stop():
#                ratio += 1/self.num_kernels
#        return ratio

