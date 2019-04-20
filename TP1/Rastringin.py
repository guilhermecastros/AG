# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:58:46 2019

@author: Guilherme
"""

import numpy as np

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def rastringin(x, n):           # F.O. Rantringin
    soma = 0.0
    for i in range(n):
        soma = soma + x[i]**2 - 10*np.cos(2*np.pi*x[i])
    f = 10*n + soma
    return f

def createNewPopulation(N):
    if N == 0:
        return [];

    population = [];
    for i in range(0, N):    
        population.append(list(np.random.choice([0, 1], size=(100,), p=[1./2, 1./2])));
    return population;

def binToDecimal(binaryArray):
   Li = -5.12;
   Ui = 5.12;
   nBits = 10;
   value = Li + (Ui - Li)*(binatyToDecimal/(2**nBits - 1));
   return value;

def evalPopulationFitness(population):
    fitness = [];
    for item in population:
        
        fitness.append(rastringin(item, 10));
    return fitness;


Li = -5.12;
Ui = 5.12;
nBits = 10;

Li = 2;
Ui = 5;
binatyToDecimal = 91;

Xi = Li + (Ui - Li)*(binatyToDecimal/(2**10 - 1));



b = np.array([0,0,0,0,0,0,0,0,0,1,1])

b.dot(2**np.arange(b.size)[::-1])