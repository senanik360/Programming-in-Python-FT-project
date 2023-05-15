import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',')

# drop unnecessary columns
df.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)

# handle missing values
df.replace(to_replace=-200, value=np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# perform exploratory data analysis
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(data=df, x='CO(GT)', kde=True, ax=axs[0][0])
axs[0][0].set_xlabel('CO(GT)')

sns.histplot(data=df, x='PT08.S1(CO)', kde=True, ax=axs[0][1])
axs[0][1].set_xlabel('PT08.S1(CO)')

sns.histplot(data=df, x='NOx(GT)', kde=True, ax=axs[1][0])
axs[1][0].set_xlabel('NOx(GT)')

sns.histplot(data=df, x='PT08.S2(NMHC)', kde=True, ax=axs[1][1])
axs[1][1].set_xlabel('PT08.S2(NMHC)')

plt.tight_layout()
plt.show()

# split the dataset into training and testing sets
X = df.drop(columns=['C6H6(GT)'])
y = df['C6H6(GT)']  # y is of type object, so sklearn cannot recognize its type
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# fit the models
nb = GaussianNB()
nb.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# predict on the test set
nb_pred = nb.predict(X_test)
knn_pred = knn.predict(X_test)
dt_pred = dt.predict(X_test)
lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)

# classification reports and accuracy scores
print('Naive Bayes:')
print(classification_report(y_test, nb_pred))
print('Accuracy:', round(accuracy_score(y_test, nb_pred), 2))
print('Confusion Matrix:')
print(confusion_matrix(y_test, nb_pred))

print('K-Nearest Neighbors:')
print(classification_report(y_test, knn_pred))
print('Accuracy:', round(accuracy_score(y_test, knn_pred), 2))
print('Confusion Matrix:')
print(confusion_matrix(y_test, knn_pred))

print('Decision Tree:')
print(classification_report(y_test, dt_pred))
print('Accuracy:', round(accuracy_score(y_test, dt_pred), 2))
print('Confusion Matrix:')
print(confusion_matrix(y_test, dt_pred))

print('Logistic Regression:')
print(classification_report(y_test, lr_pred))
print('Accuracy:', round(accuracy_score(y_test, lr_pred), 2))
print('Confusion Matrix:')
print(confusion_matrix(y_test, lr_pred))

print('Support Vector Machine:')
print(classification_report(y_test, svm_pred))
print('Accuracy:', round(accuracy_score(y_test, svm_pred), 2))
print('Confusion Matrix:')
print(confusion_matrix(y_test, svm_pred))
