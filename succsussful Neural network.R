df <- read.csv("train.csv")
df2 <- read.csv("test.csv")
df[,2:ncol(df)] <- df[2:ncol(df)]/255
zerofeature <- sapply(X = df,FUN =  max) !=0
df <- df[,c(TRUE, zerofeature)]
df$label <- as.factor(df$label)

library(RSNNS)

set.seed(123)
targetlabel <- decodeClassLabels(df[,1])
split <- splitForTrainingAndTest(df[,2:ncol(df)],
                                 y = targetlabel,
                                 ratio = .8)

nn1 <- mlp(x = split$inputsTrain, y = split$targetsTrain,
           size = c(20,20), maxit = 400,
           inputsTest = split$inputsTest,
           targetsTest = split$targetsTest)
#nn1, so far size = c(20,20) and maxit = 400 has done the best
#Accuracy : 0.9272
?mlp#

nn1.preds <- predict(nn1, split$inputsTest)
nn1.preds
nn1.preds.value <- max.col(nn1.preds) - 1
true_values <- max.col(split$targetsTest) - 1
caret::confusionMatrix(as.factor(nn1.preds.value),
                       as.factor(true_values))

features <- colnames(df[,2:length(df)])
features
feat <- paste(features)
feat <- as.list(feat)

feat
df2 <- df2/255 

summary(df2)
df2 <- df2[,colnames(df2) %in% feat]

df2
nn3.preds <- predict(nn1, df2)
nn3.preds.preds.vals <- max.col(nn3.preds)-1

summary(nn3.preds.preds.vals)
length(nn3.preds.preds.vals)
write.csv(nn3.preds.preds.vals, "submission.csv")
