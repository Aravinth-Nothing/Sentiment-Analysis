import joblib
text_model = joblib.load('Project')
value=text_model.predict(["It is the worst product ever."])
if value[0]==0:
    print('Negative')
elif value[0]==1:
    print('Positive')
import pandas as pd
df = pd.read_csv('C:/Users/aravi/Documents/Python Project/1429_1.csv', delimiter=",")
positive = 0
negative = 0
neutral = 0
print(df.columns.tolist())
df.dropna(subset=['reviews.text'], inplace=True)
for i in df['reviews.text']:
    if text_model.predict([i])[0]==2:
        positive += 1
    elif text_model.predict([i])[0] == 1:
        neutral += 1
    else:
        negative += 1  
print(positive)
print(negative)
print(neutral)
