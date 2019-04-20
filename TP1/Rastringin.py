# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:58:46 2019

@author: Guilherme
"""

import numpy as np
import random
import math

def bool2int(binaryArray):
   nArray = np.array(binaryArray)
   value = nArray.dot(2**np.arange(nArray.size)[::-1])
   return value

def binToDecimal(binaryArray):
   Li = -5.12;
   Ui = 5.12;
   nBits = 10;
   intValue = bool2int(binaryArray);
   value = Li + (Ui - Li)*(intValue/(2**nBits - 1));
   return value;

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

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

def evalPopulationFitness(population):
    fitness = [];
    variableList = [];
    for item in population:
        variablesBin = split(item, 10);
        for variable in variablesBin:
            variableList.append(binToDecimal(variable));

        fitnessValue = 1/(rastringin(variableList, 10) + 0.001);
        fitness.append(fitnessValue);
        variableList = [];
    return fitness;

def crossoverByVariable(parents):
    offspring1 = [];
    offspring2 = [];
    
    parent1Splited = split(parents[0], 10);
    parent2Splited = split(parents[1], 10);
    
    for i in range(0, len(parent1Splited)):
        pos = random.sample(range(1,9), 1)[0];
        
        offspring1.extend(parent1Splited[i][0:pos]);
        offspring1.extend(parent2Splited[i][pos:]);
        
        offspring2.extend(parent2Splited[i][0:pos]);
        offspring2.extend(parent1Splited[i][pos:]);

    return [offspring1, offspring2]

def whell(population, populationFitness):
    fitnessTot = sum(populationFitness);
    probList = [];
    for i in range(0,len(populationFitness)):
        probList.append(populationFitness[i]/fitnessTot);
        
    acumProb = np.cumsum(probList);
    randomNumber = random.sample(range(0,1000), 1)[0]/1000;
    indexFather1 = int(np.digitize(randomNumber, acumProb));

    randomNumber = random.sample(range(0,1000), 1)[0]/1000;
    indexFather2 = int(np.digitize(randomNumber, acumProb));
    
    return [population[indexFather1], population[indexFather2]];


def tournament(population, populationFitness):
    N = len(population);
    positionsSeq = list(range(0, N)) # Sequence from 0 to (N)

    # Select 10% of individuals randomly    
    numberOfIndividuals = math.floor(N * (10/100));
    positions = random.sample(positionsSeq, numberOfIndividuals);
    
    individuals = list(map(population.__getitem__, positions));
    individualsFitness = list(map(populationFitness.__getitem__, positions));
    
    individualsFitness, individuals = zip(*sorted(zip(individualsFitness, individuals)));
    
    # Return the two best individuals
    return [individuals[numberOfIndividuals - 1], individuals[numberOfIndividuals - 2]];

def bitflipMutation(child, pM):
    mutatedChild = [];
    for i in range(0,len(child)):
        if random.sample(range(0,100), 1)[0] < pM:
            mutatedChild.append(1 - child[i]);
        else:
            mutatedChild.append(child[i]);
            
    return mutatedChild;

def checkHasFoundSolution(fitnessList):
    bestSolutionFitness = 0;
    bestSolutionFitnessMax = 1/(bestSolutionFitness + 0.001);
    
    result = filter(lambda x: x == bestSolutionFitnessMax, fitnessList);
    listResult = list(result);
    return len(listResult) > 0;

#--------------------------------Main program---------------------------------#

# Number of individuals in the population
K = 100;
pM = 5; # 5%
pC = 60 # 60%
interation = 0;
population = createNewPopulation(K);
populationFitness = evalPopulationFitness(population);
checkHasFoundSolution(populationFitness);

meanFitness = [];
minFitness = [];
maxFitness = [];

while(checkHasFoundSolution(populationFitness) == False):
    interation = interation + 1;
    newPopulation = [];
    
    for i in range(0, K/2):
        if random.sample(range(0,100), 1)[0] < 50:
            selectedParents = whell(population, populationFitness);        
        else:
            selectedParents = tournament(population, populationFitness);
            
        if random.sample(range(0,100), 1)[0] < pC:    
            children = crossoverByVariable(selectedParents);
            children[0] = bitflipMutation(children[0], pM);
            children[1] = bitflipMutation(children[1], pM);
            newPopulation.extend(children);
        else:
            newPopulation.extend(selectedParents);
        
    meanFitness.append(np.mean(populationFitness));
    minFitness.append(np.min(populationFitness));
    maxFitness.append(np.max(populationFitness));
    
    
    
        

        
        
    



