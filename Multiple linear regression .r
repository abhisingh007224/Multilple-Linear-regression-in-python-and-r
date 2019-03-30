#Load the dataset
dataset = read.csv('File Name.csv')

#Convert categorical variables into numerical
  dataset$Country = factor(dataset$Country,
                       levels = c('India', 'Australia', 'NewZealand'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set

install.packages('caTools')

split = sample.split(dataset$column_name, SplitRatio = 2/3)

training_set = subset(dataset, split == TRUE)

test_set = subset(dataset, split == FALSE)


#Fitting Simple Linear Regression to the Training set and predicting test set result
regressor = lm(formula = Profit ~ .,data = training_set)

y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regressor)

