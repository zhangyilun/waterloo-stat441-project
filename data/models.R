# trying svm

setwd("4A 2015 Fall/STAT 441/project")

# library
library(e1071)

# data
X_train <- read.csv("data/X_train_dense.csv",header=T)
y_train <- read.csv("data/y_train_dense.csv",header=T) 
X_test <- read.csv("data/X_test_dense.csv",header=T)
y_test <- read.csv("data/y_test_dense.csv",header=T)

# model
model <- svm(X_train,y_train)

pred <- round(predict(model,X_test))
pred[pred == 0] <- 1
pred[pred > 7] <- 7
