import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train_clear.csv')

# KNN classifier


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

# Splitting the dataset to train and test set.

from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fitting the training data and predicting based on the test set.

knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# Evaluation


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))

error_rate_knn = []
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predict_i = knn.predict(X_test)
    error_rate_knn.append(np.mean(predict_i != y_test))

plt.plot(list(range(1, 50)), error_rate_knn)
plt.xlabel('K value')
plt.ylabel('Error rate')

# Let's choose a new K value: K=25 then refit and revaluate the model.


knn_opt = KNeighborsClassifier(n_neighbors=25)
knn_opt.fit(X_train, y_train)
pred_knn_opt = knn_opt.predict(X_test)
print(confusion_matrix(y_test, pred_knn_opt))
print(classification_report(y_test, pred_knn_opt))
