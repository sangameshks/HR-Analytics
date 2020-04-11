setwd("~/HR-Attrition-Analytics/Data")

# Loading Library ---------------------------------------------------------

library(caret)
library(randomForest)
library(dplyr)
library(rpart)
library(kenlab)
library(xgboost)
library(mltools)
library(data.table)
library(Matrix)
library(adaboost)
# Importing Data ----------------------------------------------------------

df<- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

colSums(is.na(df))
n <- nrow(df)

TrainIndex <- sample(1:n,n*.75)
# dmy <- dummyVars(as.factor(Attrition)~., data = df)
# trsf <- data.frame(predict(dmy, newdata = df))

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

ndf<- train[c("BusinessTravel","Department","EducationField","Gender",
            "JobRole","MaritalStatus","Over18","OverTime")]

mdf<- train[c("ï..Age","DailyRate","DistanceFromHome","Education","EmployeeCount"
           ,"EnvironmentSatisfaction","HourlyRate","JobInvolvement",
           "JobLevel","JobSatisfaction","NumCompaniesWorked","PercentSalaryHike",
           "PerformanceRating","RelationshipSatisfaction","StandardHours",
           "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
           "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
           "YearsSinceLastPromotion","YearsWithCurrManager","Attrition")]

oneh<- one_hot(as.data.table(ndf))
cdf<-cbind(oneh,mdf)

Attrition <- ifelse(cdf$Attrition == "No",0,1)

cdf$Attrition<- Attrition 

labels =cdf$Attrition
dtrain = xgb.DMatrix(as.matrix(sapply(cdf[,1:52], as.numeric)),
                     label= labels)
params <- list(booster = "gbtree")
xgcv_gbtree<-xgb.cv( params = params,
              data = dtrain,
              nrounds = 1000,
              nfold = 5,metrics = "error")


params <- list(booster = "dart")
xgcv_dart<-xgb.cv( params = params,
                     data = dtrain,
                     nrounds = 1000,
                     nfold = 5,metrics = "error")

params <- list(booster = "gblinear")
xgcv_gbl<-xgb.cv( params = params,
                   data = dtrain,
                   nrounds = 1000,
                   nfold = 5,metrics = "error")

min.merror.idx_linear = which.min(xgcv_gbl$evaluation_log[,test_error_mean])
min.merror.idx_gbtree = which.min(xgcv_gbtree$evaluation_log[,test_error_mean])
min.merror.idx_dart = which.min(xgcv_dart$evaluation_log[,test_error_mean])


ndf<- test[c("BusinessTravel","Department","EducationField","Gender",
              "JobRole","MaritalStatus","Over18","OverTime")]

mdf<- test[c("ï..Age","DailyRate","DistanceFromHome","Education","EmployeeCount"
              ,"EnvironmentSatisfaction","HourlyRate","JobInvolvement",
              "JobLevel","JobSatisfaction","NumCompaniesWorked","PercentSalaryHike",
              "PerformanceRating","RelationshipSatisfaction","StandardHours",
              "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
              "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
              "YearsSinceLastPromotion","YearsWithCurrManager","Attrition")]

oneh<- one_hot(as.data.table(ndf))
cdf<-cbind(oneh,mdf)
Attrition <- ifelse(cdf$Attrition == "No",0,1)

cdf$Attrition<- Attrition 

labels =cdf$Attrition
dtest = xgb.DMatrix(as.matrix(sapply(cdf[,1:52], as.numeric)),
                     label= labels)

y<-as.matrix(as.integer(cdf$Attrition))

bst <- xgboost(param=params, data=dtrain, label=y, 
               nrounds=min.merror.idx_linear, verbose=0)

d<- predict(bst,newdata= dtest, type = "response")
quantile(d)
gg1=floor(d+0.5)

confusionMatrix(as.factor(gg1),as.factor(y))


