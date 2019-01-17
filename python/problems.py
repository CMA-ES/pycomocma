#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Cheikh
"""

import numpy as np

class BiobjectiveConvexQuadraticProblem(object):
    """
    """
    def __init__(self,
                 dim,
                 hessian_first = None,
                 hessian_second = None,
                 optimum_first = None,
                 optimum_second = None,
                 scaling_first = None,
                 scaling_second = None,
                 name = None):
        self.dim = dim
        self.optimum_first = optimum_first if optimum_first else np.zeros(self.dim)
        self.optimum_second = optimum_second if optimum_second else np.ones(self.dim)
        
        if hessian_first:
            self.hessian_first = hessian_first
        if hessian_second:
            self.hessian_second = hessian_second
            
        if name == "sphere":
            self.hessian_first = np.eye(self.dim,self.dim)
            self.hessian_second = np.eye(self.dim,self.dim)
            
        elif name == "elli":
            self.hessian_first = np.eye(self.dim,self.dim)
            self.hessian_second = np.eye(self.dim,self.dim)
            if self.dim > 1:
                for i in range(self.dim):
                    self.hessian_first[i,i] = 1e6**(i/self.dim-1)
                    self.hessian_second[i,i] = 1e6**(i/self.dim-1)
                
        elif name == "cigtab":
            self.hessian_first = np.eye(self.dim,self.dim)
            self.hessian_second = np.eye(self.dim,self.dim)
            self.hessian_first[0,0] = 10**-4
            self.hessian_second[0,0] = 10**-4
            if self.dim > 1:
                self.hessian_first[1,1] = 10**4
                self.hessian_second[1,1] = 10**4
            
        else: #choose two random orthogonal matrices
            B = np.random.randn(self.dim, self.dim)
            for i in range(self.dim):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.hessian_first = hessian_first if hessian_first else B
            self.hessian_second = hessian_second if hessian_second else B
        
        Q1,Q2 = self.hessian_first, self.hessian_second
        x1,x2 = self.optimum_first, self.optimum_second
      
        scale = max(np.dot((x2-x1).T, np.dot(Q1,x2-x1)), np.dot(
                (x1-x2).T, np.dot(Q2,x1-x2)) )

        self.scaling_first = scaling_first if scaling_first else scale
        self.scaling_second = scaling_second if scaling_second else scale
        
        self.name = name


    def objective_functions(self):
        """
        """
        x1 = self.optimum_first
        x2 = self.optimum_second
        def fun1(x):
            x = np.array(x)
            return 1/self.scaling_first*np.dot((x-x1).T, np.dot(
                    self.hessian_first,x-x1) )
        def fun2(x):
            x = np.array(x)
            return 1/self.scaling_second*np.dot((x-x2).T, np.dot(
                    self.hessian_second,x-x2) )
        return fun1, fun2
    
    #put also Peter Bosman functions, multi-modals ...
    def sep(self,k,O = False, Two_O = False):
        
        """
        self.sep(k) is sep-k
        if O = True, then sep-O
        if Two_O = True, then sep-Two-O
        """
        self.optimum_first = np.zeros(self.dim)
     #   self.name = self.name + "_sep{}".format(k)
        if not O:
            self.optimum_second = np.zeros(self.dim)
            self.optimum_second[k] = 1
        else:
    #        self.name = self.name + "_o"
            B = np.random.randn(self.dim, self.dim)
            for i in range(self.dim):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.optimum_second = np.dot(B,np.ones(self.dim))
            if Two_O:
       #         self.name = self.name + "_Two_o"                
                C = np.random.randn(self.dim, self.dim)
                for i in range(self.dim):
                    for j in range(0, i):
                        C[i] -= np.dot(C[i], C[j]) * C[j]
                    C[i] /= sum(C[i]**2)**0.5
                self.hessian_second = np.dot(C.T, np.dot(self.hessian_second, C))
            
            
        Q1,Q2 = self.hessian_first, self.hessian_second
        x1,x2 = self.optimum_first, self.optimum_second
      
        scale = max(np.dot((x2-x1).T, np.dot(Q1,x2-x1)), np.dot(
                (x1-x2).T, np.dot(Q2,x1-x2)) )

        self.scaling_first = scale
        self.scaling_second = scale
        
    def one(self,O = False):
        """
        """
        self.optimum_first = np.zeros(self.dim)
   #     self.name = self.name + "_one"

        if not O:
            self.optimum_second = np.ones(self.dim)            
        else:
    #        self.name = self.name + "_o"            
            B = np.random.randn(self.dim, self.dim)
            for i in range(self.dim):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.optimum_second = np.dot(B,np.ones(self.dim))
        
  
            
   
        C = np.random.randn(self.dim, self.dim)
        for i in range(self.dim):
            for j in range(0, i):
                C[i] -= np.dot(C[i], C[j]) * C[j]
            C[i] /= sum(C[i]**2)**0.5
        self.hessian_first = np.dot(C.T, np.dot(self.hessian_first, C))
        self.hessian_second = np.dot(C.T, np.dot(self.hessian_second, C))
  
       
        Q1,Q2 = self.hessian_first, self.hessian_second
        x1,x2 = self.optimum_first, self.optimum_second
      
        scale = max(np.dot((x2-x1).T, np.dot(Q1,x2-x1)), np.dot(
                (x1-x2).T, np.dot(Q2,x1-x2)) )

        self.scaling_first = scale
        self.scaling_second = scale
   
          
      
    def two(self,O = False):
        """
        """
        self.optimum_first = np.zeros(self.dim)
        if not O:
            self.optimum_second = np.ones(self.dim)            
        else:
            B = np.random.randn(self.dim, self.dim)
            for i in range(self.dim):
                for j in range(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.optimum_second = np.dot(B,np.ones(self.dim))
   
        C = np.random.randn(self.dim, self.dim)
        for i in range(self.dim):
            for j in range(0, i):
                C[i] -= np.dot(C[i], C[j]) * C[j]
            C[i] /= sum(C[i]**2)**0.5

        D = np.random.randn(self.dim, self.dim)
        for i in range(self.dim):
            for j in range(0, i):
                D[i] -= np.dot(D[i], D[j]) * D[j]
            D[i] /= sum(D[i]**2)**0.5
            
        self.hessian_first = np.dot(C.T, np.dot(self.hessian_first, C))
        self.hessian_second = np.dot(D.T, np.dot(self.hessian_second, D))
        
        Q1,Q2 = self.hessian_first, self.hessian_second
        x1,x2 = self.optimum_first, self.optimum_second
      
        scale = max(np.dot((x2-x1).T, np.dot(Q1,x2-x1)), np.dot(
                (x1-x2).T, np.dot(Q2,x1-x2)) )

        self.scaling_first = scale
        self.scaling_second = scale
   
     
