# Machine-Learning-Hiring

Exercise



This file contains hiring statics for a firm such as experience of candidate, his written test score and personal interview score. Based on these 3 factors, HR will decide the salary. Given this data, you need to build a machine learning model for HR department that can help them decide salaries for future candidates. Using this predict salaries for following candidates,



2 yr experience, 9 test score, 6 interview score

12 yr experience, 10 test score, 10 interview score





!pip install word2number

import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv(r'C:/Users/Jaydeep Patel/Downloads/hiring.csv')
df

df.info()

df.experience = df.experience.fillna('zero')
df

df.experience = df.experience.apply(w2n.word_to_num)
df

import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
df

model = linear_model.LinearRegression()
model.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

model.predict([[2,9,6]])

model.predict([[12,10,10]])

model.coef_

model.intercept_

2922.26901502*12+2221.30909959*10+2147.48256637*10+14992.65144669314

