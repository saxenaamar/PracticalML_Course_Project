## Coursera - Practical Machine Learning - John Hopkins Data Science Specialization
## Course Project 


## Load libraries
library(readr)
library(dplyr)
library(caret)
library(parallel)
library(doParallel)
library(kernlab)



## Load Data
training <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <-read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")


## Data Cleaning

### Remove all the variables having only NA as their value
testing<-testing[,colSums(is.na(testing))<nrow(testing)]
litemp<-c(colnames(testing)[1:59],"classe")
training<-training %>% select(litemp)
rm(litemp)

### Variables with missing values
colnames(training)[colSums(is.na(training))>0]

### Number of NAs in these variables
sum(is.na(training$magnet_dumbbell_z))
sum(is.na(training$magnet_forearm_y))
sum(is.na(training$magnet_forearm_z))

### Row Index for these variables
which(is.na(training),arr.ind=TRUE)

### Dropping the row - as it is only 1 row - compared to total of 19,622 records
training<-training[-c(5373),]

### Selecting only the numeric variables for building model
numeric<-lapply(training,is.numeric)
numeric$classe<-TRUE
numeric$X1<-FALSE
numeric$user_name<-FALSE
numeric$raw_timestamp_part_1<-FALSE
numeric$raw_timestamp_part_2<-FALSE
numeric$cvtd_timestamp<-FALSE
numeric$num_window<-FALSE
numericCol<-unlist(numeric)
training<-training[,numericCol]
testing<-testing[,numericCol]
colnames(testing)[53]<-"classe"
rm(numeric)

### Split data into training (70%) and testing (30%)
set.seed(1)
inTrain<-createDataPartition(y=training$classe,p=0.7,list=FALSE)
training_sub<-training[inTrain,]
testing_sub<-training[-inTrain,]

### Using parallel implementation to increase speed of Random Forest implementation
### And afterwards stopping it
cluster <-makeCluster(detectCores()-1)
registerDoParallel(cluster)

### Build model on training dataset
set.seed(1)
# model 1: SVM
modFit1<-train(classe~.,method="svmRadial",data=training_sub)
# model 2: GBM
modFit2<-train(classe~.,method="gbm",data=training_sub)
# model 3: random forest
modFit3<-train(classe~.,method="rf",data=training_sub)

fitControl<-trainControl(method="cv",number=20,allowParallel = TRUE)
modFit<-train(classe~.,method="rf",data=training_sub,trControl=fitControl)


stopCluster(cluster)
registerDoSEQ()


### Find accuracy of prediction
ConMatrix<-confusionMatrix(table(testing_sub$classe,predict(modFit,testing_sub)))
ConMatrix

### Prediction
pred<-predict(modFit,testing)

### Importance of Features
importance<-varImp(modFit,scale=FALSE)
plot(importance)
