## import dependencies
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing,cross_validation
import pandas as pd

## reading the dataset present in excel format
df=pd.read_excel('titanic.xls')
#print (df.head())

## drop name column
df.drop(['body','name'],1,inplace=True)

## encode object variables
df.convert_objects(convert_numeric=True)

## fill NAs
df.fillna(0,inplace=True)


def handle_non_numeric_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]= x
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
    return df


df = handle_non_numeric_data(df)
#print (df.head())

## convert data to array of floats
X=np.array(df.drop(['survived'],1).astype(float))

## scaling data
X=preprocessing.scale(X)

## predicted values
y=np.array(df['survived'])

## KNN classifier
clf=KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct+=1
print(correct/len(X))   ##accuracy
