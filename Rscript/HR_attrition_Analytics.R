setwd("~/HR-Attrition-Analytics/Data")

# Loading Library ---------------------------------------------------------

library(caret)
library(randomForest)
library(dplyr)
library(rpart)


# Importing Data ----------------------------------------------------------

df<- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

colSums(is.na(df))
n <- nrow(df)

TrainIndex <- sample(1:n,n*.75)
dmy <- dummyVars(as.factor(Attrition)~., data = df)
trsf <- data.frame(predict(dmy, newdata = df))

train <- df[TrainIndex,]
test <- df[-TrainIndex,]

rprt_model<-rpart(as.factor(Attrition)~.,data = train)
rprt_predict<- predict(rprt_model,newdata=test,type="class")
confusionMatrix(rprt_predict,test$Attrition)

rand_model <- randomForest(Attrition~.,data = train)

train.control <- trainControl(method="cv", number=100)
# Train the model
model <- train(Attrition~.,data = train, method = "rpart",
               trControl = train.control)
