library('class')
library('caret')
library('mlbench')

evalK <- function(K, dataX, dataY, foldList) {
  
  X <- dataX
  Y <- dataY

  #foldList <- list()
  #foldList <- createFolds(1:nrow(X), 10)
  #foldList <- folds
  
  results <- rep(0,length(foldList))
  
  for(i in c(1:length(foldList))){
    
    iFold <- 1:length(foldList)
    #filtrar a lista
    iFold <- which(iFold != i)
    DataTrain <- rep(0,0)
    for(j in c(1:length(iFold)))
    {
      iDataTrain <- c(DataTrain, foldList[[iFold[j]]]) 
    }
    
    DataTrain <- X[iDataTrain,]
    DTrain <- Y[iDataTrain]
    DataTest <- X[foldList[[i]],]
    DTest <- Y[foldList[[i]]]
    
    tempResult <- knn(DataTrain,DataTest,DTrain,K)
    
    results[i] <- (sum(1*(tempResult == DTest))/length(DTest))
  }
  
  meanResult <- mean(results)
  
 return (meanResult)
}