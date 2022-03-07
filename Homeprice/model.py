
# salary prediction model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
home=pd.read_csv("homepricess.csv")
x=home.drop(['price'],axis=1)
y=home.drop(['area'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3,random_state=32)
#since the data set is too small we are taking the whole data to train model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#fitting the model
m=regressor.fit(x_train,y_train)
#saving the model to disk
pickle.dump(regressor,open('model.pkl','wb'))