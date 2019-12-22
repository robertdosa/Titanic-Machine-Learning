#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# create a Pandas data frame from the train set.

train = pd.read_csv('train.csv')

# # Data preprocessing

train = train.drop(['Cabin', 'Name', 'Ticket'], axis=1)


def impute_age(cols):
    """
    cols: DataFrame object
    Returns mean age for each Pclass if the value is NaN, else returns Age.
    """
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return np.mean(train[train['Pclass'] == 1]['Age'])
        elif Pclass == 2:
            return np.mean(train[train['Pclass'] == 2]['Age'])
        else:
            return np.mean(train[train['Pclass'] == 3]['Age'])
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# Encoding categorical data and updating the dataframe


sex = pd.get_dummies(train['Sex'], drop_first=True)  # Creating a new DataFrame with dummy variables for the sex column
emb = pd.get_dummies(train['Embarked'],
                     drop_first=True)  # Creating a new DataFrame with dummy variables for the embarked column
train = pd.concat([train, sex, emb], axis=1)  # Concatenating the datasets
train = train.drop(['Sex', 'Embarked'], axis=1)  # Dropping remaining sex and embarked columns

# # Classification with random forest classifier


from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200, random_state=11)

rfc.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report

pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))

errorRateDepth = []
depthList = list(range(1, 10))
for i in depthList:
    rfcOpt = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=11)
    rfcOpt.fit(X_train, y_train)
    predOpt = rfcOpt.predict(X_test)
    errorRateDepth.append(np.mean(y_test != predOpt))

plt.figure(figsize=(12, 6))
plt.plot(depthList, errorRateDepth)
plt.xlabel('Max Depth')
plt.ylabel('Error rate')

rfc_opt = RandomForestClassifier(n_estimators=200, max_depth=3)
rfc_opt.fit(X_train, y_train)
pred_rfc_opt = rfc_opt.predict(X_test)
print(confusion_matrix(y_test, pred_rfc_opt))
print(classification_report(y_test, pred_rfc_opt))

# This time we could reach a slightly better accuracy of 0.85.
