#importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the dataset
dataset=pd.read_csv('File name.csv')
X= dataset.iloc[:,:-3].values # X is a independent variable
Y= dataset.iloc[:,3].values         # y is a dependent variable


 
#Encoding categorical data into numerical if required
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()      
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features =[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
    X=X[:,1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=0)

#prediction on test set result
  y_pred = reg.predict(X_test)

# Buiding the optimal model using backward elimination

    import statsmodels.formula.api as sm 
    X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) 
    X_opt=X[:,[0,1,2,3,4,5]] 
    regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 
    regressor_OLS.summary()

#    If p is greater than significance level remove that predictor
 #   if it is less than significance level it stays in models.    