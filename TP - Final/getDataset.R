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
library('readr')
library('DMwR')

getDataset <- function(name) {
  
  if(name == "diabetes")
  {
    data("diabetes")
    X <- data.matrix(diabetes[,2:4])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y[which(diabetes$class == 'Normal'),1] <- 1
    Y[which(diabetes$class == 'Chemical'),1] <- -1
    Y[which(diabetes$class == 'Overt'),1] <- -1
    
    xc <- X
  }
  else if(name == "BreastCancer")
  {
    data(BreastCancer)
    X <- data.matrix(BreastCancer[,2:10])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y[which(BreastCancer$Class == 'benign'),1] <- 1
    Y[which(BreastCancer$Class == 'malignant'),1] <- -1

    xc <- X
  }
  else if(name == "iris")
  {
    data(iris)
    X <- data.matrix(iris[,1:4])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y[which(iris$Species == 'setosa'),1] <- 1
    Y[which(iris$Species != 'setosa'),1] <- -1
    
    xc <- X
  }
  else if(name == 'Ionosphere')
  {
    data(Ionosphere)
    X <- data.matrix(Ionosphere[,3:34])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y[which(Ionosphere$Class == 'good'),1] <- 1
    Y[which(Ionosphere$Class != 'good'),1] <- -1
    
    xc <- X
  }
  else if(name == 'Sonar')
  {
    data(Sonar)
    X <- data.matrix(Sonar[,1:60])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y[which(Sonar$Class == 'R'),1] <- 1
    Y[which(Sonar$Class != 'R'),1] <- -1
    
    xc <- X
  }
  else if(name == 'voting')
  {
    data <- data.matrix(read.csv("voting", header = FALSE))
    X <- data.matrix(data[,2:ncol(data)])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y <- as.numeric(as.factor(data.matrix(data[,1])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == 'liver disosters')
  {
    data <- read.csv("liver", header = FALSE)
    X <- data.matrix(data[,1:ncol(data)-1])
    X[is.na(X)] <- 0
    Y <- matrix(0, nrow = dim(X)[1], ncol = 1)
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == 'PimaIndiansDiabetes')
  {
    data(PimaIndiansDiabetes)
    X <- data.matrix(PimaIndiansDiabetes[,1:ncol(PimaIndiansDiabetes)-1])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)]))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == 'wine')
  {
    data <- data.matrix(read.csv("wine", header = FALSE))
    X <- data.matrix(data[,2:ncol(data)])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,1])))
    Y[which( Y == 2)] <- -1
    Y[which( Y == 3)] <- -1
    
    xc <- X
  }
  else if(name == "fertility_Diagnosis")
  {
    data <- read.table("fertility_Diagnosis.txt", sep = ",")
    X <- data.matrix(data[,1:ncol(data) - 1])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == "heart")
  {
    data <- read.csv("heart", header = FALSE, sep = " ")
    X <- data.matrix(data[,1:ncol(data)-1])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == "hepatitis")
  {
    data <- read.table("hepatitis.txt", sep = ",", stringsAsFactors=FALSE)
    X <- data.matrix(data[,2:ncol(data)])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,1])))
    Y[which( Y == 2)] <- -1
    
    for(i in 1:nrow(X))
    {
      for(j in 1:ncol(X))
      {
        if(X[i,j] == "?")
        {
          X[i,j] <- 0
        }
      }
    }

    xc <- X
  }
  else if(name == "planning_relax")
  {
    data <- read.table("planning_relax.txt")
    X <- data.matrix(data[,1:ncol(data) - 1])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == "tic-tac-toe")
  {
    data <- read.csv("tic-tac-toe", header = FALSE)
    X <- data[,1:ncol(data)-1]
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  else if(name == "haberman")
  {
    data <- read.csv("haberman", header = FALSE)
    X <- data.matrix(data[,1:ncol(data)-1])
    X[is.na(X)] <- 0
    Y <- as.numeric(as.factor(data.matrix(data[,ncol(data)])))
    Y[which( Y == 2)] <- -1
    
    xc <- X
  }
  
  

  return (list(xc,Y))
}