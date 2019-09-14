rm(list=ls())
library('bmp')
library('rgl')
library('caret')
library('mlbench')
library('RSNNS')
library('mclust')
library('plot3D')
library('e1071')
library('class')
library('caTools')
source('createFirstCostGenearation.R')
source('createFirstGammaGenearation.R')
source('calcFitness.R')
source('getDataset.R')
library('readr')
library('DMwR')

# 1 - diabetes
# 2 - BreastCancer
# 3 - iris
# 4 - Ionosphere
# 5 - Sonar
# 6 - voting
# 7 - liver disosters
# 8 - PimaIndiansDiabetes
# 9 - wine
# 10 - fertility_Diagnosis
# 11 - heart
# 12- hepatitis
# 13 - planning_relax
# 14 - tic-tac-toe
# 15 - haberman
data <- getDataset("fertility_Diagnosis")
xc <- data[[1]]
Y <- data[[2]]
resultAcc <- rep(0,30)
iterations <- 0
while(iterations <= 30)
{
  message(iterations)
  iterations <- iterations + 1 
  index <- sample(dim(xc)[1])
  
  dimXTrain <- round(dim(xc)[1]*0.6)
  dimXTest <- round(dim(xc)[1]*0.2)
  dimXValidation <- round(dim(xc)[1]*0.2)
  
  XTest <- xc[index[1:dimXTest],]
  DTest <- Y[index[1:dimXTest]]
  
  XTrain <- xc[index[(dimXTest + 1): (dimXTrain + dimXTest)],]
  DTrain <- Y[index[(dimXTest + 1): (dimXTrain + dimXTest)]]
  
  XValidation <- xc[index[((dimXTest + dimXTrain) + 1): dim(xc)[1]],1:ncol(xc)]
  DValidation <- Y[index[((dimXTest + dimXTrain) + 1): dim(xc)[1]]]
  
  DTest <- as.factor(DTest)
  DTrain <- as.factor(DTrain)
  DValidation <- as.factor(DValidation)
  
  cost <- c(2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11,2^12,2^13,2^14,2^15)
  gamma <- c(2^-15,2^-14,2^-13,2^-12,2^-11,2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3)
  
  costBin <- rep(0, length(cost))
  gammaBin <- rep(0, length(gamma))
  
  #Probability of crossOver
  probCrossOver <- 0.7
  
  #Number of children in each generation
  numOfChild <- 2
  
  #Probability of mutation
  probMutation <- 0.7
  
  #N is the number of population
  N <- 20
  
  costFather <- createFirstCostGenearation(N, length(cost))
  gammaFather <- createFirstGammaGenearation(N, length(gamma))
  fathersFitness <- rep(0, N)
  
  #number of generations
  k <- 0
  
  #number of generations whithout change the bestFitness
  bestChange <- 0 
  
  bestIndex <- 0
  bestCost <- 0
  bestGamma <- 0
  bestAcc <- 0
  
  for(i in (1: N))
  {
    #Eval fitness of each father
    fathersFitness[i] <- calcFitness(costFather[i,], gammaFather[i,], DTrain, XTrain, XValidation, DValidation)
  }
  
  while(k < 300 || bestChange < 10){
  
    if(max(fathersFitness) > bestAcc)
    {
      bestIndex <- which(fathersFitness == max(fathersFitness))[1]
      bestCost <- cost[which(costFather[bestIndex,] == 1)]
      bestGamma <- gamma[which(gammaFather[bestIndex,] == 1)]
      bestAcc <- max(fathersFitness)
      bestChange <- 0
    }
    else
    {
      bestChange <- bestChange + 1
    }
    numberOfChildren <- 0
    while(numberOfChildren < numOfChild)
    {
      numberOfChildren <- numberOfChildren + 1
      #Eval crossover
      if(sample(1:10, 1) <= probCrossOver*10)
      {
        
        if(sample(1:10, 1) <= 5)
        {
          #the children gets the cost from father1
          index1 <- which(fathersFitness == max(fathersFitness))[1]
          childrenCost <- costFather[index1,]
          
          #and the gamma from father2
          index2 <- which(fathersFitness == sort(fathersFitness)[N-1])[1]
          childrenGamma <- gammaFather[index2,]
        }
        else
        {
          #the children gets the cost from father1
          index1 <- which(fathersFitness == sort(fathersFitness)[N-1])[1]
          childrenCost <- costFather[index1,]
          
          #and the gamma from father2
          index2 <- which(fathersFitness == max(fathersFitness))[1]
          childrenGamma <- gammaFather[index2,]
        }
        
        #Eval mutation
        if(sample(1:10, 1) <= probMutation*10)
        {
          # First estrutura de vizinhança
          
          #childrenCost <- rep(0, length(cost))
          #childrenGamma <- rep(0, length(gamma))
          #childrenCost[sample(1:length(cost),1)] <- 1
          #childrenGamma[sample(1:length(gamma),1)] <- 1
          
          #Second estrutura de vizinhança
          indexGamma <- which(childrenGamma == 1)
          indexCost <-  which(childrenCost == 1)
          
          newIndexGamma <- sample(1:length(gamma),1) + indexGamma
          newIndexCost <- sample(1:length(cost),1) + indexCost
          
          if(newIndexGamma > length(gamma))
          {
            newIndexGamma <- newIndexGamma - length(gamma)
          }
          
          if(newIndexCost > length(cost))
          {
            newIndexCost <- newIndexCost - length(cost)
          }
          
          childrenCost <- rep(0, length(cost))
          childrenGamma <- rep(0, length(gamma))
          
          childrenCost[newIndexCost] <- 1
          childrenGamma[newIndexGamma] <- 1
        }
        
        childrenFitness <- calcFitness(childrenCost, childrenGamma, DTrain, XTrain, XValidation, DValidation)
        if(childrenFitness > min(fathersFitness))
        {
          substituteIndex <- which(fathersFitness == min(fathersFitness))[1]
          costFather[substituteIndex,] <- childrenCost
          gammaFather[substituteIndex,] <- childrenGamma
          fathersFitness[substituteIndex] <- childrenFitness
        }
      }
    }
    k <- k + 1
  }
  #print("Best Accuracy Validation")
  #print(bestAcc)
  resultTest <- calcFitness(costFather[bestIndex,], gammaFather[bestIndex,], DTrain, XTrain, XTest, DTest)
  #print("Best Accuracy Test")
  #print(resultTest)
  resultAcc[iterations] <- resultTest
}
mean(resultAcc)