# Script used to model the Machine Learning Week 4 project

## Load the files from appropriate directories
library(caret)
relUrlT <- "~/Week4/pml-training.csv"
relUrlE <- "~/Week4/pml-testing.csv"
trainD <- read.csv(relUrlT,header = TRUE)
validationD <- read.csv(relUrlE, header = TRUE)   ## will be used as out of sample data to validate

## Set seed and partition the training data into 
set.seed(12345)
inTrain <- createDataPartition(y=trainD$classe, p=0.7, list = FALSE)
training <- trainD[inTrain,]
testing <- trainD[-inTrain,]

## DATA CLEANUP
## To elimnate the columns with no variance that will be a weak predictors in the modeling
## Applying the same structural change (removal of fields) across the data sets, including validation

trainNZV <- nearZeroVar(training)
training <- training[,-trainNZV]
testing <- testing[,-trainNZV]
validation <- validationD[,-trainNZV]

## Remove NA filled columns (this is discretionary)
naColumns <- sapply(names(training), function(x) all(is.na(training[,x]) == TRUE))
training <- training[, naColumns==FALSE]
testing <- testing[, naColumns==FALSE]
validation <- validation[,naColumns == FALSE]

## MODELING & CROSS-VALIDATION AND MODEL SELCTION BASED ON ACCURACY RESULTS

trCtrl <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelFit_rpart <- train(classe ~ ., method = "rpart", data = na.exclude(training), trControl = trCtrl)
modelFit_rf <- train(classe ~ ., method = "rf", data = na.exclude(training), prox=TRUE, trControl = trCtrl)
modelFit_gbm <- train(classe ~ ., method = "gbm", data = na.exclude(training), verbose= FALSE , trControl = trCtrl)


## PREDICTION OF TESTING AND VALIDATION DATASET USING THE SELECTED MODEL (GBM BASED)
pred_rpart <- predict(modelFit_rpart,newdata = na.exclude(testing))
pred_rf <- predict(modelFit_rf,newdata = na.exclude(testing))
pred_gbm <- predict(modelFit_gbm,newdata = na.exclude(testing))

## ACCURACY RESULTS WITH AND W/O NA REMOVAL IN THE BASE DATA SETS
print(mean(modelFit_gbm$results$Accuracy))  # [1] 0.9975757 ,[1] 0.9983897
print(mean(modelFit_rpart$results$Accuracy)) # [1] 0.5601901, [1] 0.6065754
print(mean(modelFit_rf$results$Accuracy)) # [1] 0.9256475, [1] 0.9365079

pred_oos <- predict(modelFit_gbm,newdata = na.exclude(validation))



