library('class')
library('caret')
library('mlbench')
library('e1071')

createFirstCostGenearation <- function(numberOfPopulation, numberOfValues) {
  
  costMatriz <- matrix(0, numberOfPopulation, numberOfValues)
  
  for(i in (1: numberOfPopulation))
  {
    #generate randon number
    costMatriz[i,sample(1:numberOfValues,1)] <- 1
  }
  
  return (costMatriz)
}