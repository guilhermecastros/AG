library('class')
library('caret')
library('mlbench')
library('e1071')

evalByAcc <- function(dataX, dataY, foldList, cost, gamma) {
  
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
    
    svm.model <- svm(DTrain ~ ., data = DataTrain, cost = cost, gamma = gamma)
    svm.pred <- predict(svm.model, DataTest)

    Pxc <- as.numeric(as.character(svm.pred))

    #results[i] <- colAUC(svm.pred,DTest, plotROC=FALSE)
    results[i] <- (sum(1*(DTest == Pxc))/length(Pxc))
  }
  
  meanResult <- mean(results)
  
  return (meanResult)
}