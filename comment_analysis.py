import pandas as pd
import numpy as np
df=pd.read_excel("C:/Users/aravi/Documents/Python Project/ScrubbedDataCorrected.xlsx")
df["Comments"].fillna(value = "", inplace = True)
df.dropna(subset=['Rating', 'Comments'], inplace=True)
summary=[]
for i in df['Rating']:
  if i>3:
    summary.append(1)
  else:
    summary.append(0)
x=df['Comments'].values.astype('U')
y=pd.DataFrame(summary)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
model=SVC()
from sklearn.pipeline import make_pipeline
text_model=make_pipeline(CountVectorizer(),SVC())
text_model.fit(x_train,y_train)
y_pred=text_model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
import joblib
joblib.dump(text_model,'Project')
import joblib
text_model=joblib.load('Project')
print(text_model.predict(["It was really bad"]))
print(text_model.predict(["It was the best lens i have ever used!"]))



 

