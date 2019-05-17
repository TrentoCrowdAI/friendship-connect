from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import pandas

import matplotlib.pyplot as plt

def import_training_data():
  """
  Import the dataset from the information source, that contains the known connection
  between pairs, to be used as training data.
    
  Each row in the dataset represents whether the pairs (user and friend) shared 
  a certain aspect in common. These "common aspects" are the predictors.
  
  The second column in the dataset is the known level of connectedness.    
  
  This example imports the data from a CSV file. If you want to import it from a 
  database, look at the following link: 
  https://datatofish.com/sql-to-pandas-dataframe/  
  
  """
  
  dataset = pandas.read_csv('data/pairs-training.csv')
  
  # Y is the value you want to predict, in this case the level of connectedness
  # this line is only grabbing the second column in the dataset
  y = dataset["connectedness_level"].values
  
  # X contains all the predictors, meaning the features that we are going to use
  # to predict Y. In this case, this is all the information except the second column
  x = dataset.drop(columns = "connectedness_level").values  
  
  return x, y

def prepare_model(x, y):   
  """
  We use a multiple linear regression model to predict the connectedness
  between two participants.
  x is the list of predictors
  y is the dependent variable (connectedness)
  """
  # fits a linear regression model with the y(connecteness) and x(predictors)
  lm = LinearRegression()
  lm.fit(x, y)
  
  return lm

def predict_unknown_frienships(lm):
  """
  This function predicts the friendships we don't know anything about, meaning 
  that we don't have the "connectedness" level for them.    
  """
  # First you compute the aspects in common between the new or unknown pairs
  # we have this information on a CSV file, but you probably will connect to 
  # a database
  dataset = pandas.read_csv('data/pairs-unknown.csv')
  
  # In our dataset we have the actual value of connectedness (y), so we get rid of it
  x = dataset.drop(columns = "connectedness_level").values  
  # I keep the true value of y so we can make comparisons later
  yTrue = dataset["connectedness_level"].values
  
  yPredicted = lm.predict(x)
  
  # In real settings you would like to update the database with the predicted values
  # saveToDatabase(yPredicted)
  data = {'yTrue' : yTrue, 'yPredicted' : yPredicted}
  df = pandas.DataFrame(data)
  df.to_csv("predicted.csv")
  
  return yPredicted, yTrue

## Main
x, y = import_training_data()
model = prepare_model(x, y)
yPredicted, yTrue = predict_unknown_frienships(model)



plt.scatter(yTrue, yPredicted, color = 'red')
plt.xlabel("Connectedness (true value)")
plt.ylabel("Connectedness (predicted)")
plt.savefig("plot-true_vs_predicted.png")

