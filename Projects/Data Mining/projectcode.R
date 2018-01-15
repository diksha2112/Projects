install.packages("randomForest")
library(randomForest)

data=read.csv("parkinsons.csv")
data=data.frame(data)
data$status=as.factor(data$status)
y=data[98:195,1]
data_train=data.frame(data[1:97,1:24])
data_train$status=as.factor(data_train$status)
data_test=data.frame(data[98:195,2:24])
# Fitting model
fit =randomForest(status ~ ., data_train,ntree=500)
summary(fit)
#Predict Output 
predicted=predict(fit,data_test)
predicted
TP=0
TN=0
FP=0
FN=0

for (i in 1:98){
  if(data[97+i,1]==predicted[i]){
    if(data[97+i,1]==1){
    TP=TP+1}
    else{
      TN=TN+1
    }
  }
  if(data[97+i,1]!=predicted[i])
  {
    if(data[97+i,1]==0){
    FP=FP+1
    }
    else{
      FN=FN+1
    }
    }

}

accuracy=(TP+TN)/98
#confusion matrix
CM=matrix(c(TP,FN,FP,TN),nrow=2,ncol=2,byrow=true)
rownames(CM)=c("Actual 1","Acttual 0")
colnames(CM)=c("Predicted1","Predicted 0")

plot(fit)
varImpPlot(fit,
           main="Variable Importance",
           n.var=5)
library(ggplot2)
qplot(PPE,spread1,colour=status,data=data)
qplot(data[,5],data[,9],colour=status,data=data,main="MDVP Flo Hz(x) Vs MDVP PPQ(y)")
qplot(data[,6],spread1,colour=status,data=data,main="MDVP Jitter(x) Vs spread1(y)")
qplot(data[,6],data[,5],colour=status,data=data,main="MDVP Jitter(x) Vs MDVP Flo Hz(y)")
qplot(PPE,data[,5],colour=status,data=data,main="MDVP Flo Hz(x) Vs PPE(y)")