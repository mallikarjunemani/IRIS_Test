# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = sns.load_dataset('iris')

X = data[data.columns[:-1]]

y = data[data.columns[-1]]

# plt.xlabel('Features')
# plt.ylabel('Species')

# pltX = data.loc[:, 'sepal_length']
# pltY = data.loc[:,'species']
# plt.scatter(pltX, pltY, color='blue', label='sepal_length')

# pltX = data.loc[:, 'sepal_width']
# pltY = data.loc[:,'species']
# plt.scatter(pltX, pltY, color='green', label='sepal_width')

# pltX = data.loc[:, 'petal_length']
# pltY = data.loc[:,'species']
# plt.scatter(pltX, pltY, color='red', label='petal_length')

# pltX = data.loc[:, 'petal_width']
# pltY = data.loc[:,'species']
# plt.scatter(pltX, pltY, color='black', label='petal_width')

# plt.legend(loc=4, prop={'size':8})
# plt.show()

from sklearn import preprocessing


lr = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = preprocessing.scale(X_train)

X_test_scaled = preprocessing.scale(X_test)

lr.fit(X_train_scaled, Y_train)

y_pred = lr.predict(X_test_scaled)

print('LR Coefficient {}'.format(lr.coef_))

print('Classification Report {}'.format(classification_report(Y_test, y_pred)))

print('Accuracy Score {}'.format(accuracy_score(Y_test, y_pred)))