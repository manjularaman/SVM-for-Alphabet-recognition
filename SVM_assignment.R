############################ SVM Number Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear Kerrnel
#  4.3 Polynomial Kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

#####################################################################################

# 2. Data Understanding: 

# Train Data Number of Instances: 59999
# Train Data Number of Attributes: 785
# Test Data Number of Instances: 9999
# Test Data Number of Attributes: 785

#3. Data Preparation: 


#Loading Neccessary libraries
install.packages("parallel")
install.packages("caret")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("gridExtra")
library(parallel)
library(kernlab)
library(readr)
library(caret)

#Loading Data

Data1 <- read_csv("mnist_train.csv")
Data2 <- read_csv("mnist_test.csv")

#Understanding Dimensions

dim(Data1)
dim(Data2)

#Structure of the dataset

str(Data1)
str(Data2)


#checking missing value

sapply(Data1, function(x) sum(is.na(x)))
sapply(Data2, function(x) sum(is.na(x)))

any(is.na(Data1))
any(is.na(Data2))
#any(duplicated(Data1))
#any(duplicated(Data2))

#Changing column names
colnames(Data1)[1] <- c("label")
for(i in 2:ncol(Data1)) {colnames(Data1)[i] <- paste("X",i,sep="")}
colnames(Data2)[1] <- c("label")
for(i in 2:ncol(Data2)) {colnames(Data2)[i] <- paste("X",i,sep="")}

###Taking 15% of DATA for training and test
set.seed(1)
train.indices = sample(1:nrow(Data1), 0.15*nrow(Data1))
train = Data1[train.indices, ]
set.seed(2)
test.indices = sample(1:nrow(Data2),0.15*nrow(Data2))
test = Data2[test.indices, ]

#Making our target class to factor
train$label <- factor(train$label)
test$label <- factor(test$label)

#Constructing Model

############################################Using Linear Kernel##############################################
Model_linear <- ksvm(label~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$label)
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.98485   0.9940  0.93038  0.89474   0.9202  0.83688  0.94286  0.90741  0.81955  0.88742
#Specificity           0.99488   0.9947  0.98583  0.98293   0.9888  0.98454  0.99485  0.99327  0.99341  0.99036
#Accuracy : 0.9133 
Model_linear10 <- ksvm(label~ ., data = train, scale = FALSE, kernel = "vanilladot",C=10)
Eval_linear10<- predict(Model_linear10, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear10,test$label)
#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.98485   0.9940  0.93038  0.89474   0.9202  0.83688  0.94286  0.90741  0.81955  0.88742
#Specificity           0.99488   0.9947  0.98583  0.98293   0.9888  0.98454  0.99485  0.99327  0.99341  0.99036
#Accuracy :  0.9133
Model_linear5 <- ksvm(label~ ., data = train, scale = FALSE, kernel = "vanilladot",C=5)
Eval_linear5<- predict(Model_linear5, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear5,test$label)
#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.98485   0.9940  0.93038  0.89474   0.9202  0.83688  0.94286  0.90741  0.81955  0.88742
#Specificity           0.99488   0.9947  0.98583  0.98293   0.9888  0.98454  0.99485  0.99327  0.99341  0.99036
#Accuracy : 0.9133  

###Values of acuuracy , sensitivities and specificity on the test data are same for models trained with C=1,5,10 

#####################################################################
# Hyperparameter tuning and Cross Validation  - Linear - SVM 
######################################################################

# We will use the train function from caret package to perform crossvalidation

trainControl <- trainControl(method="cv", number=3)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"
set.seed(100)
# making a grid of C values. 
grid <- expand.grid(C=seq(0.10,2, by=0.1))

# Performing 3-fold cross validation
fit.svm <- train(label~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm)

# Plotting "fit.svm" results
plot(fit.svm)
####Results after 3 fold cross validation
#C    Accuracy   Kappa    
#0.1  0.9079904  0.8976944
#0.2  0.9079904  0.8976944
#0.3  0.9079904  0.8976944
#0.4  0.9079904  0.8976944
#0.5  0.9079904  0.8976944
#0.6  0.9079904  0.8976944
#0.7  0.9079904  0.8976944
#0.8  0.9079904  0.8976944
#0.9  0.9079904  0.8976944
#1.0  0.9079904  0.8976944
#1.1  0.9079904  0.8976944
#1.2  0.9079904  0.8976944
#1.3  0.9079904  0.8976944
#1.4  0.9079904  0.8976944
#1.5  0.9079904  0.8976944
#1.6  0.9079904  0.8976944
#1.7  0.9079904  0.8976944
#1.8  0.9079904  0.8976944
#1.9  0.9079904  0.8976944
#2.0  0.9079904  0.8976944

###Below were the resuls using the 5 fold cross validation
##For C values between 
#0.01,0.02,0.03,0.04,0.05, Accuracy = 0.908101
#0.05 and 0.25 Accuracy = 0.908101
#0.30 and 0.55 Accuracy = 0.908101
#1,2,3,4,5 Accuracy= 0.908101
#20,30,40,50,60 Accuracy = 0.908101 
#70,90,110,130,150,170  Accuracy = 0.908101
#200,300,400,500,600 Accuracy = 0.908101

###the linear model accuracies are not changing with the C value; 

#Final linear model with C=0.1
Eval_linear_final<- predict(fit.svm, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear_final,test$label)
#########Final accuracy for linear model is 0.9133###
###############################################################################
###############################################Using Polynomial Kernel to get better Accuracy############################
Model_Poly <- ksvm(label~ ., data = train, scale = FALSE, kernel = "polydot")
Eval_Poly <- predict(Model_Poly,test)
#confusion matrix - Poly Kernel
confusionMatrix(Eval_Poly,test$label)
#Accuracy : 0.9133
####degree 2 polynomial trialModel
Model_Poly <- ksvm(label~ ., data = train, scale = FALSE, kernel = "polydot",kpar(degree=2,C=1))
Eval_Poly <- predict(Model_Poly,test)
#confusion matrix - Poly Kernel
confusionMatrix(Eval_Poly,test$label)
#Accuracy : 0.9653
#####################################################################
# Hyperparameter tuning and Cross Validation  - Poly - SVM 
######################################################################
trainControl <- trainControl(method="cv", number=2)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
#Using C (1,2) and small values of sigma
set.seed(7)
grid <- expand.grid(.degree=c(1,2,3), .C=c(0.1,0.5,0.7),.scale=c(1) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmPoly", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)
#degree  C    Accuracy   Kappa    
#1       0.1  0.9047671  0.8941116
#1       0.5  0.9047671  0.8941116
#1       0.7  0.9047671  0.8941116
#2       0.1  0.9458831  0.9398301
#2       0.5  0.9458831  0.9398301
#2       0.7  0.9458831  0.9398301
#3       0.1  0.9395493  0.9327817
#3       0.5  0.9395493  0.9327817
#3       0.7  0.9395493  0.9327817
#The final values used for the model were degree = 2, scale = 1 and C = 0.1.
Eval_Poly_final <- predict(fit.svm,test)
#confusion matrix - Poly Kernel
confusionMatrix(Eval_Poly_final,test$label)

#Final Accuracy on polynomial model is Accuracy : 0.9653  
################################################Using RBF Kernel to get better Accuracy##################################
Model_RBF <- ksvm(label~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$label)
#Accuracy : 0.96
#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.98485   0.9940  0.93671  0.95395   0.9509  0.94326  0.97143   0.9568  0.93985   0.9669
#Specificity           0.99707   0.9985  0.99553  0.99480   0.9955  0.99558  0.99411   0.9955  0.99634   0.9926

############   Hyperparameter tuning and Cross Validation #####################
#for speedup 
allow_parallel = "true"
# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=2)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
#Using C (1,2) and small values of sigma
set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(1,2) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)

#For C values 1,2 and sigma 0.025 and 0.05, the accuracy is very low 0.1174575 
#sigma  C  Accuracy   Kappa
#0.025  1  0.1174575  0    
#0.025  2  0.1174575  0    
#0.050  1  0.1174575  0    
#0.050  2  0.1174575  0 


#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 0.05 and C = 1.

#Modeling with C values 0.1 and 0.5 and sigma=0.5
#C     Accuracy   Kappa
#0.05  0.1174575  0    
#0.10  0.1174575  0 



###Model_RBF have 96% accuracy; Clearly hyperparameter is very low  Hyperparameter : sigma =  1.63535123814415e-07 


trainControl <- trainControl(method="cv", number=2)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
#Using C (1) and small values of sigma (2e-7,1e-6)
set.seed(7)
grid <- expand.grid(.sigma=c(2e-7,1e-6), .C=c(1) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)

#sigma  Accuracy   Kappa    
#2e-07  0.9463278  0.9403279
#1e-06  0.9399937  0.9333019

###Trying a few values less that 2e-7

set.seed(7)
grid <- expand.grid(.sigma=c(1e-7,9e-8), .C=c(1) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)
#9e-08  0.9317706  0.9241380
#1e-07  0.9336598  0.9262389

###accuracy decreasing as the sigma decreases to 1e-7 ; increasing sigma above 2e-7

set.seed(7)
grid <- expand.grid(.sigma=c(3e-7,5e-7, 7e-7), .C=c(1) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
#for speedup 
allow_parallel = "true"

print(fit.svm)

plot(fit.svm)
# sigma  Accuracy   Kappa    
# 3e-07  0.9513283  0.9458881
# 5e-07  0.9531063  0.9478663
# 7e-07  0.9522174  0.9468791
#The highest accuracy value is around sigma = 5e-07 
#Using this value for the test dataset 
################################################Using RBF Kernel##################################

Eval_RBF_final<- predict(fit.svm, test)
Eval_RBF_final
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_final,test$label)
 ##overall accuracy was 96%

####Using this value of sigma, altering the C to find a better accuracy.

set.seed(7)
grid <- expand.grid(.sigma=c(5e-7), .C=c(0.1,0.5) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
#for speedup 
allow_parallel = "true"

print(fit.svm)

plot(fit.svm)
#C    Accuracy   Kappa    
#0.1  0.9073239  0.8969725
#0.5  0.9476611  0.9418124

###Accuracy 94.7% less than 96%
#Trying values 0.7 and 1.2

set.seed(7)
grid <- expand.grid(.sigma=c(5e-7), .C=c(0.7,1.2) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
#for speedup 
allow_parallel = "true"

print(fit.svm)

plot(fit.svm)
#C    Accuracy   Kappa    
#0.7  0.9502170  0.9446540
#1.2  0.9534396  0.9482366

###Accuracy increases Changing C between 1.5 to 2.5

set.seed(7)
grid <- expand.grid(.sigma=c(5e-7), .C=c(1.5,2,2.5) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
#for speedup 
allow_parallel = "true"

print(fit.svm)

plot(fit.svm)

#C    Accuracy   Kappa    
#1.5  0.9546620  0.9495958
#2.0  0.9554399  0.9504604
#2.5  0.9552176  0.9502132
###Most optimal C seems to be 2.0
####Final RBF model with C=2.0 ans sigma = 5e-7
set.seed(7)
grid <- expand.grid(.sigma=c(5e-7), .C=c(2) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(label~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
#for speedup 
allow_parallel = "true"

print(fit.svm)


Eval_RBF_final_with_c<- predict(fit.svm, test)
Eval_RBF_final_with_c
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_final_with_c,test$label)
##overall accuracy was 97.2%

##################################Conclusion####################################
#Interms of Accuracy, the best model is RBF with sigma = 5e-7 and c=2; Final accuracy = 97.2%
##Polynomial model of degree 2, c=0.1 gives second best accuracy 96.53%
##Linear model accuracy was 91.33%


