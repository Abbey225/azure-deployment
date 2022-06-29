import numpy as np
import pandas as pd

df=pd.read_csv('car_dataset.csv')
del df['Unnamed: 0']
print(df.head())

x=df.drop('selling_price',axis=1)
y=df.selling_price

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.ensemble import ExtraTreesRegressor
et_model=ExtraTreesRegressor()

et_model.fit(x_train,y_train)

print(et_model.score(x_train,y_train))
print(et_model.score(x_test,y_test))

import pickle
pickle.dump(et_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
