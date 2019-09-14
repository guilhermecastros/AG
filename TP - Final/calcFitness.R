library('class')
library('caret')
library('mlbench')
library('e1071')

calcFitness  <- function (costFather, gammaFather, DTrain, XTrain, XTest, DTest) {
  
  costValues <- c(2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11,2^12,2^13,2^14,2^15)
  gammaValues <- c(2^-15,2^-14,2^-13,2^-12,2^-11,2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3)
  
  cost <- costValues[which(gammaFather == 1)]
  gamma <- gammaValues[which(gammaFather == 1)]
  
  svm.model <- svm(DTrain ~ ., data = XTrain, cost = cost, gamma = gamma)
  svm.pred <- predict(svm.model, XTest)
  
  Pxc <- as.numeric(as.character(svm.pred))
  #print((sum(1*(DTest == Pxc))/length(Pxc)))
  
  acc <- (sum(1*(DTest == Pxc))/length(Pxc))
  
  return (acc)
}