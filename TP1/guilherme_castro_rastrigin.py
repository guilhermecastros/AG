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

def grayToBinary(grayArray):
    binArray = [];
    binArray.append(grayArray[0]);
    
    for i in range(1, len(grayArray)):
        if binArray[i-1] == grayArray[i]:
            binArray.append(0);
        else:
            binArray.append(1);
            
    return binArray;

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

def rastrigin(x, n):           # F.O. Rantringin
    soma = 0.0
    for i in range(n):
        soma = soma + x[i]**2 - 10*np.cos(2*np.pi*x[i])
    f = 10*n + soma
    return f

def createNewPopulation(N, nvar):
    if N == 0:
        return [];

    population = [];
    for i in range(0, N):    
        population.append(list(np.random.choice([0, 1], size=(10*nvar,), p=[1./2, 1./2])));
    return population;

def getIndividualVariables(item):
    variableList = [];
    variablesBin = split(item, 10);
    for variable in variablesBin:
        variableBin = grayToBinary(variable);
        variableList.append(binToDecimal(variableBin));
        #variableList.append(binToDecimal(variable));
        
    return variableList;

def getIndividualRealFitness(item, nvar):
    variableList = getIndividualVariables(item);
    fitnessValue = rastrigin(variableList, nvar);
    
    return fitnessValue;

def evalPopulationFitness(population, nvar):
    fitness = [];
    variableList = [];
    for item in population:
        variablesBin = split(item, 10);
        for variable in variablesBin:
            variableBin = grayToBinary(variable);
            variableList.append(binToDecimal(variableBin));
            #variableList.append(binToDecimal(variable));

        fitnessValue = 1/(rastrigin(variableList, nvar) + 0.001);
        #fitnessValue = rastrigin(variableList, 10);
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
    #return [individuals[0], individuals[1]];


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

def guilherme_castro(nvar, ncal):
    # Number of individuals in the population
    K = 50;
    pM = 2; # 5%
    pC = 60; # 60%
    interation = 0;
    population = createNewPopulation(K, nvar);
    populationFitness = evalPopulationFitness(population, nvar);
    checkHasFoundSolution(populationFitness);
    
    bestSolution = population[np.argmax(populationFitness)];
    bestSolutionFitness = np.max(populationFitness);
    bestSolutionRealFitness = getIndividualRealFitness(bestSolution, nvar);
    
    meanFitness = [];
    minFitness = [];
    maxFitness = [];
    
    meanFitness.append(np.mean(populationFitness));
    minFitness.append(np.min(populationFitness));
    maxFitness.append(np.max(populationFitness));
    
    avaliation = 0;
    
    while((checkHasFoundSolution(populationFitness) == False) & (avaliation < ncal)):
        #print(interation);
        interation = interation + 1;
        newPopulation = [];
        newPopulationFitness = [];
        selectedParents = [];
        
        for i in range(0, 10):
            if random.sample(range(0,100), 1)[0] < 50:
                selectedParents = whell(population, populationFitness);
            else:
                selectedParents = tournament(population, populationFitness);
                
            if random.sample(range(0,100), 1)[0] < pC:
                children = crossoverByVariable(selectedParents);
                children[0] = bitflipMutation(children[0], pM);
                children[1] = bitflipMutation(children[1], pM);
                newPopulation.extend(children);
                avaliation = avaliation + 2;
            #else:
            #    newPopulation.extend(selectedParents);
                
        #population = newPopulation;
        newPopulationFitness = evalPopulationFitness(newPopulation, nvar);
    
        population.extend(newPopulation);
        populationFitness.extend(newPopulationFitness);
        
        populationFitness, population = zip(*sorted(zip(populationFitness, population)));
        
        population = list(population[len(newPopulation):]);
        populationFitness = list(populationFitness[len(newPopulation):]);
        
        if bestSolutionFitness < np.max(populationFitness):
            bestSolution = population[np.argmax(populationFitness)];
            bestSolutionFitness = np.max(populationFitness);
            bestSolutionRealFitness = getIndividualRealFitness(bestSolution, nvar);
            
        meanFitness.append(np.mean(populationFitness));
        minFitness.append(np.min(populationFitness));
        maxFitness.append(np.max(populationFitness));
        
    return [bestSolution, bestSolutionRealFitness]
    
    
    
        

        
        
    



