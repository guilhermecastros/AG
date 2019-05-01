# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
# Solution example
# [5,1,8,4,2,7,3,6]

def fitnessNQueens(solution):
    # returns the amount of collisions for a given solution (permutation).
    # the maximum amount of collisions that can occur is n(n-1)/2.
    # for a 4x4 chessboard, the maximum amount of collisions is 6, corresponding 
    # to the situation in which all queens are in the same diagonal.
    f = 0;
    n = len(solution);
    for i in range(n):
        for j in range(n):
            if (abs(i-j) == abs(solution[i]-solution[j])) & (i!=j):
                f=f+1;               
    f=f/2;
    return f;

def evalPopulationFitness(population):
    fitness = [];
    for item in population:
        fitness.append(fitnessNQueens(item));
    return fitness;

def createNewPopulation(N):
    if N == 0:
        return [];

    population = [];
    # Sequence from 1 to 8
    rangeList = list(range(1, 9));
    for i in range(0, N):
        # Return a k length list of unique elements chosen from the population sequence.
        # Used for random sampling without replacement.    
        population.append(random.sample(rangeList, 8));
    return population;

def checkHasFoundSolution(fitnessList):
    result = filter(lambda x: x == 0, fitnessList);
    listResult = list(result);
    return len(listResult) > 0;

def mutation(child):
    positionsSeq = list(range(0, len(child))) # Sequence from 0 to N
    # Get two positions without repetition
    position1, position2 = random.sample(positionsSeq, 2);
    
    mutatedChild = child;
    
    aux = mutatedChild[position1];
    mutatedChild[position1] = mutatedChild[position2];
    mutatedChild[position2] = aux;
    
    return mutatedChild;

def cutAndCrossFillCrossover(parents):
    N = len(parents[0]);
    offspring1 = [0]*N;
    offspring2 = [0]*N;
    
    positionsSeq = list(range(1, N)) # Sequence from 1 to (N)
    # Get one positions without repetition
    pos = random.sample(positionsSeq, 1)[0];

    offspring1[0:pos] = parents[0][0:pos];
    offspring2[0:pos] = parents[1][0:pos];
    s1 = pos;
    s2 = pos;
    for i in range(N):
        check1 = 0;
        check2 = 0;
        for j in range(pos):
            if parents[1][i] == offspring1[j]:
                check1 = 1;
            if parents[0][i] == offspring2[j]:
                check2 = 1;

        if check1 == 0:
            offspring1[s1] = parents[1][i];
            s1 = s1 + 1;
        if check2 == 0:
            offspring2[s2] = parents[0][i];
            s2 = s2 + 1;

    return [offspring1, offspring2]

def selectParents(population, populationFitness):
    N = len(population);
    positionsSeq = list(range(0, N)) # Sequence from 0 to (N)

    # Select 5% of individuals randomly    
    numberOfIndividuals = math.floor(N * (5/100));
    positions = random.sample(positionsSeq, numberOfIndividuals);
    
    individuals = list(map(population.__getitem__, positions));
    individualsFitness = list(map(populationFitness.__getitem__, positions));
    
    individualsFitness, individuals = zip(*sorted(zip(individualsFitness, individuals)));
    
    # Return the two best individuals
    return [individuals[0], individuals[1]];
    
#--------------------------------Main program---------------------------------#

def guilherme_castro():
    # Number of individuals in the population
    K = 100;
    pM = 10; # 60%
    interation = 0;
    population = createNewPopulation(K);
    populationFitness = evalPopulationFitness(population);
    checkHasFoundSolution(populationFitness);
    
    meanFitness = [];
    minFitness = [];
    maxFitness = [];
    
    while((checkHasFoundSolution(populationFitness) == False) & (interation != (5000))):
        interation = interation + 1;
        selectedParents = selectParents(population, populationFitness);
        children = cutAndCrossFillCrossover(selectedParents);
        
        if random.sample(range(0,100), 1)[0] < pM:
            children[0] = mutation(children[0]);
            
        if random.sample(range(0,100), 1)[0] < pM:
            children[1] = mutation(children[1]);
        
        childrenFitness = evalPopulationFitness(children);
        
        population.extend(children);
        populationFitness.extend(childrenFitness);
        
        populationFitness, population = zip(*sorted(zip(populationFitness, population)));
        
        population = list(population[0:K]);
        populationFitness = list(populationFitness[0:K]);
        
        meanFitness.append(np.mean(populationFitness));
        minFitness.append(np.min(populationFitness));
        maxFitness.append(np.max(populationFitness));
        
    return [population[np.argmin(populationFitness)], np.min(populationFitness)];
    
    
    #plt.plot(meanFitness,label = 'Colisões Médio')
    #plt.plot(minFitness,label = 'Colisões Mínimo')
    #plt.plot(maxFitness,label = 'Colisões Máximo')
    #plt.legend()
    #plt.show()




