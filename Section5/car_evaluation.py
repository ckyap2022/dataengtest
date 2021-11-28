#Import the necessary modules from specific libraries.

import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import export_graphviz 

#Load the data set

df = pd.read_csv('car_data.csv',names=['buying','maint','doors','persons','lug_boot','safety','class'])
df.shape
df.head()

# filter dataframe based on the requirement
data = df[(df['maint'] == "high") | (df['doors'] == "4") | (df['lug_boot'] == "big") | (df['safety'] == "high") | (df['class'] == "good")]

#Taking an overview of data
data.sample(10)

#Let's check if there are any missing values in our dataset 
data.isnull().sum()

#We see that there are no missing values in our dataset 
#Let's take a more analytical look at our dataset 
data.describe()

#We realize that our data has categorical values 
data.columns

mt = pd.crosstab(data['buying'],  data['maint'])
drs = pd.crosstab(data['buying'],  data['doors'])
lb = pd.crosstab(data['buying'],  data['lug_boot'])
sfty = pd.crosstab(data['buying'],  data['safety'])
cls = pd.crosstab(data['buying'],  data['class'])

def predict_buying_price(desc, tdf):
	# select all rows by : and column 1
	# by 1:2 representing features
	X = buy.iloc[:, 1:2].astype(int) 
  
	# print X
	# print(X)

	# select all rows by : and column 2
	# by 2 to Y representing labels
	y = tdf.iloc[:, 2].astype(int) 
  
	# print y
	# print(y)

	# import the regressor
	from sklearn.tree import DecisionTreeRegressor 
  
	# create a regressor object
	regressor = DecisionTreeRegressor(random_state = 0) 
  
	#Train test split, split data randomly into 70% training and 30% test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	# fit the regressor with X and Y data
	regressor.fit(X_train, y_train)
 
	# test the output using training data
	y_pred = regressor.predict(X_test)
  
	# print the predicted price
	print("Predicted price: for " + desc + " = % d\n"% y_pred[0]) 


predict_buying_price("maintainance", mt)
predict_buying_price("doors", drs)
predict_buying_price("lug_boot", lb)
predict_buying_price("safety", sfty)
predict_buying_price("class", cls)
