library(caret)
relUrlT <- "~/coursera/datascience/MachineLearning/Week4/pml-training.csv"
relUrlE <- "~/coursera/datascience/MachineLearning/Week4/pml-testing.csv"
trainD <- read.csv(relUrlT,header = TRUE)
validationD <- read.csv(relUrlE, header = TRUE)


set.seed(12345)
inTrain <- createDataPartition(y=trainD$classe, p=0.7, list = FALSE)
training <- trainD[inTrain,]
testing <- trainD[-inTrain,]

trainNZV <- nearZeroVar(training)
#testingNZV <- nearZeroVar(testing)
training <- training[,-trainNZV]
testing <- testing[,-trainNZV]
validation <- validationD[,-trainNZV]


naColumns <- sapply(names(training), function(x) all(is.na(training[,x]) == TRUE))
training <- training[, naColumns==FALSE]
testing <- testing[, naColumns==FALSE]
naColumns <- sapply(names(validation), function(x) all(is.na(validation[,x]) == TRUE))
validation <- validation[,naColumns == FALSE]

trCtrl <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelFit_rpart <- train(classe ~ ., method = "rpart", data = na.exclude(training), trControl = trCtrl)
modelFit_rf <- train(classe ~ ., method = "rf", data = na.exclude(training), prox=TRUE, trControl = trCtrl)
modelFit_gbm <- train(classe ~ ., method = "gbm", data = na.exclude(training), verbose= FALSE , trControl = trCtrl)

pred_rpart <- predict(modelFit_rpart,newdata = na.exclude(testing))
pred_rf <- predict(modelFit_rf,newdata = na.exclude(testing))
pred_gbm <- predict(modelFit_gbm,newdata = na.exclude(testing))
print(mean(modelFit_gbm$results$Accuracy))  # [1] 0.9975757 ,[1] 0.9983897
print(mean(modelFit_rpart$results$Accuracy)) # [1] 0.5601901, [1] 0.6065754
print(mean(modelFit_rf$results$Accuracy)) # [1] 0.9256475, [1] 0.9365079
pred_oos <- predict(modelFit_gbm,newdata = na.exclude(validation))


