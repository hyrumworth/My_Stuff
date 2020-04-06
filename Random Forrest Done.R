install.packages("randomForest")
library(randomForest)
df <- read.csv("train.csv")

df[,2:ncol(df)] <- df[,2:ncol(df)]/255
df$label <- as.factor(df$label)

#work with nearZeroVar to get rid of those zeros
#may be able to get to the low 500 if you use this 
#to the extreeme 
zero_feature <- sapply(df[,2:ncol(df)], max) != 0
zero_feature <- c(TRUE, zero_feature)
#nearZeroVar(df,)
df <- df[,zero_feature]
summary(train1[,1:3])

#next split intno train and validation
set.seed(54321)
inTrain <- createDataPartition(df$label, p = .05, list = F)
train <- df[inTrain,]
test <- df[-inTrain,]
#don't forget to optimize the random forest
rf1 <- randomForest(y = train[,1], 
                    train[,2:ncol(train)],
                    ntree = 50)
rf1
here 
predictions_rf <- predict(rf1, test)
confusionMatrix(predictions_rf, test$label)

validate <- read.csv("test.csv", stringsAsFactors = T)
validate <- validate[,zero_feature]

rf_kaggle <- predict(rf1, validate)
?write.csv
submission <- write.csv(rf_kaggle, "submission.csv") 
#expected for your project understand what a 
#confusion matrix is telling you.