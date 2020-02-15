```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

a=[20,23,25,26,29,30,31,33,37,20,21,25,26,27,30,31,34,35]
b=[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
c=['Hiphop','Hiphop','Hiphop','Jazz','Jazz','Jazz','Classical','Classical','Classical','Dance','Dance','Dance','Acoustic','Acoustic','Acoustic','Classical','Classical','Classical']
dataframe= pd.DataFrame({'age':a,'gender':b,'genre':c})
dataframe.to_csv("music.csv",index=False,sep=',')

model= DecisionTreeClassifier()
model.fit(X,y)
predictions=model.predict([[21,1],[22,0]])
predictions
```




    array(['Hiphop', 'Dance'], dtype=object)



# split into a training set and a testing set


```python
music_data=pd.read_csv('music.csv')
X=music_data.drop(columns=['genre'])
y=music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


model= DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
predictions=model.predict(X_test)

score= accuracy_score(y_test,predictions)
score
```




    1.0



# persisting a model


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
# music_data=pd.read_csv('music.csv')
# X=music_data.drop(columns=['genre'])
# y=music_data['genre']

# model= DecisionTreeClassifier()
# model.fit(X,y)

joblib.dump(model,'music-recommender.joblib')

#predictions=model.predict([[21,1]])
```




    ['music-recommender.joblib']




```python
model=joblib.load('music-recommender.joblib')
predictions=model.predict([[21,1]])
predictions
```




    array(['Hiphop'], dtype=object)



# visualizing a decision tree


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data=pd.read_csv('music.csv')
X=music_data.drop(columns=['genre'])
y=music_data['genre']

model= DecisionTreeClassifier()
model.fit(X,y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age','gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)
```


```python

```
