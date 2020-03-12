install.packages("caret")
install.packages("e1071")
install.packages("randomForest")
install.packages("kernlab")
install.packages("dplyr")
install.packages("manip")

library(caret)
library(randomForest)
library(kernlab)
library(dplyr)

train <- read.csv("train.csv" , stringsAsFactors = F)
#lastvalidation <- read.csv("test.csv",stringsAsFactors = F)
fix_embarked <- function(df){
  df$Embarked <- sub("", "B", df$Embarked)
  df$Embarked <- sub("BS", "S", df$Embarked)
  df$Embarked <- sub("BC", "C", df$Embarked)
  df$Embarked <- sub("BQ", "Q", df$Embarked)  
}


summary(train)
inTrain <-createDataPartition(y = train$Survived, p = .80, list = FALSE)

training <- train[inTrain,]
testing <- train[-inTrain,]
fix_Survived <- function(df){
  df <- df%>%
    mutate(NSurvived = ifelse(Survived == 1, "yes", "no"))
  return(df)
}
training <- fix_Survived(training)
testing <- fix_Survived(testing)

#fix_Embarked <- function(df){
 # df <- df%>%
  #  mutate(NEmbarked = ifelse(Embarked == "NA", "C", Embarked))
  #return(df)
#}

#training <- fix_Embarked(training)
#testing <- fix_Embarked(testing)
str(training)
str(testing)
summary(training)
summary(testing)
training <- training[,c("Sex", "Pclass","Embarked", "NSurvived")]
testing <- testing[,c("Sex", "Pclass","Embarked", "NSurvived")]

makefactors <- function(df1){
  df1$Sex <- as.factor(df1$Sex)
  df1$NEmbarked <- as.factor(df1$Embarked)
  df1$Pclass <- as.factor(df1$Pclass)
  df1$NSurvived <- as.factor(df1$NSurvived)  
  return(df1)
}

training <- makefactors(training)
testing <- makefactors(testing)

summary(training)


train_control <- trainControl(method = "CV",
                              number = 5,
                              savePredictions = T,
                              classProbs = T)
modelFitGLM <- train(NSurvived ~., data = training, 
                     trControl = train_control, 
                     method = "glm")
modelFitRF <- train(NSurvived ~., data = training, 
                    trControl = train_control, 
                    method = "rf")
modelFitLB <- train(NSurvived ~., data = training, 
                    trControl = train_control, 
                    method = "LogitBoost")
modelFitLDA <- train(NSurvived ~., data = training, 
                     trControl = train_control, 
                     method = "lda")
set.seed(54321)
#modelFitRF <- train(type ~., data = training, 
#trControl = train_control, 
#method = "rf")

predictGLM <- predict(modelFitGLM, newdata = testing)
predictRF <- predict(modelFitRF, newdata = testing)
predictLDA <- predict(modelFitLDA, newdata = testing)
predictLB <- predict(modelFitLB, newdata = testing)

confusionMatrix(predictGLM, testing$NSurvived)
confusionMatrix(predictRF, testing$NSurvived) 
confusionMatrix(predictLDA, testing$NSurvived)
confusionMatrix(predictLB, testing$NSurvived)

####REcombine data ####
finaltrain <- read.csv("train.csv", stringsAsFactors = F)
kaggletest <- read.csv("test.csv", stringsAsFactors = F)

fix_embarked(finaltrain)
fix_embarked(kaggletest)

head(kaggletest)

#finaltrain <- fix_Embarked(finaltrain)
#kaggletest <- fix_Embarked(kaggletest)
finaltrain <- fix_Survived(finaltrain) 

finaltrain <- makefactors(finaltrain)
makefactors2 <- function(df1){
  df1$Sex <- as.factor(df1$Sex)
  df1$Embarked <- as.factor(df1$Embarked)
  df1$Pclass <- as.factor(df1$Pclass)
  return(df1)
}
kaggletest <- makefactors2(kaggletest)


finaltrain2 <- finaltrain[,c("Sex", "Pclass","Embarked", "NSurvived")]
kaggletest2 <- kaggletest[,c("Sex", "Pclass","Embarked")]

makefactors(finaltrain2)
makefactors2(kaggletest2)
head(finaltrain2)
head(kaggletest2)

str(finaltrain2)
str(kaggletest2)
summary(finaltrain2)

train_control <- trainControl(method = "CV",
                              number = 5,
                              savePredictions = T,
                              classProbs = T)
modelFitRF2 <- train(NSurvived ~., data = finaltrain2, 
                     trControl = train_control, 
                     method = "rf")


predictRF2 <- predict(modelFitRF2, newdata = kaggletest)

predictRF2

submission <- cbind(kaggletest$PassengerId, predictRF2)
submission <- data.frame(submission)


submission <-submission%>%
  mutate(submission$predictRF2, Survived = ifelse(submission$predictRF2 == 1,0,1))
submission$PassengerId <- submission$V1
names(submission)
submission <- submission[,c("PassengerId", "Survived")]
submission
write.csv(submission, "submission.csv", 
          row.names = FALSE)

####step 7 communicate results #### 
