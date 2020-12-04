# This is my number prediction code for the kaggle competition. 
####set up####
install.packages("e1071")
install.packages("caret")
library(e1071)
library(caret)

####Import data####
train <- read.csv("train.csv", stringsAsFactors = F)
finalset <- read.csv("test.csv", stringsAsFactors = F)

####data exploration####
summary(train$label)
str(train)
train$label <- as.factor(train$label)
####data split####
set.seed(5301)
datapar <- createDataPartition(train$label, times = 1, p = .05, list = F)
training <- train[datapar,]
testing <- train[datapar,]
####support vector machine####
svm_model <- svm(label ~., data = training,
                 type = "C-classification",
                 kernel = "polynomial",
                 degree = 2,
                 scale = T,
                 gamma = .1,
                 coef0 = .1)

summary(svm_model)
pred_train <- predict(svm_model, training)
mean(pred_train == training$label)
#100% on the training stuff now for the testing set
pred_test <- predict(svm_model, testing)
mean(pred_test == testing$label)
summary(pred_test)
####SVM submission to kaggle####
final_svm <- predict(svm_model, finalset)
Label <- final_svm
summary(final_svm)
ImageId <- 1:28000
combined_vectors <- as.data.frame(cbind(ImageId, Label))
summary(final_svm)
write.csv(Label, file = "submission2.csv")
?write.csv
sparce <- createData
####tuning the svm####
tuni_out <- tune.svm(x = training[,-1],
                     y = training[,1], 
                     type = "C-classification", 
                     kernel = "polynomial", 
                     degree = 2,
                     gamma = c(.1,1,10),
                     coef0 = c(.1,1,10))

summary(tuni_out)

####Neural Network####
#setup
library(neuralnet)
train <-read.csv("train.csv")

zero_feature <- sapply(train[,2:ncol(train)], max) != 0
zero_feature <- c(TRUE, zero_feature)
#nearZeroVar(df,)
train <- train[,zero_feature]
summary(train1[,1:3])

maxs <- apply(train[,2:ncol(train)],2, max)
mins <- apply(train[,2:ncol(train)],2, min)
maxs
mins
scaled.data <- as.data.frame(scale(train[,2:ncol(train)],
                             center = mins,
                             scale = maxs - mins))

summary(maxs)
summary(mins)
summary(scaled.data)

scaled.data[colSums(scaled.data),]
scaled.data <- cbind(as.numeric(train$label),
                     scaled.data)
#scaled.data[is.na(scaled.data)] <- 0
summary(scaled.data)
colnames(scaled.data)[1] <- "Label"
summary(scaled.data)
trainIndex <- createDataPartition(scaled.data$Label, p = .05, list = F)
train2 <- scaled.data[trainIndex,]
test2 <- scaled.data[-trainIndex,]

colnames(scaled.data[,2:length(scaled.data)])
head(scaled.data)
features <- colnames(scaled.data[,2:length(scaled.data)])
f <- paste(features, collapse = "+")
f <- paste("Label ~", f)
f
summary(train2)
summary(train2)
set.seed(53012)
nn <- neuralnet(formula = f, data = train2, hidden = c(10,10),
                     linear.output = F, err.fct = "sse", stepmax = 1e6)
plot(nn)
three.three.prediction <- compute(nn, test2)
head(three.three.prediction$net.result)
three.three.prediction

levels(test)
three.three.output <- sapply(three.three.prediction$net.result, round, digits = 0)

table(three.three.output, test2$Label)
confusionMatrix(as.factor(three.three.output), as.factor(test2$label))


j <- predict(nn, finalset)
summary(j)
submission <- write.csv(j, "submission.csv")
