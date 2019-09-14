library('class')
library('caret')
library('mlbench')
library('e1071')

createFirstGammaGenearation <- function(numberOfPopulation, numberOfValues) {
  
  gammaMatriz <- matrix(0, numberOfPopulation, numberOfValues)
  
  for(i in (1: numberOfPopulation))
  {
    #generate randon number
    gammaMatriz[i,sample(1:numberOfValues,1)] <- 1
  }
  
  return (gammaMatriz)
}