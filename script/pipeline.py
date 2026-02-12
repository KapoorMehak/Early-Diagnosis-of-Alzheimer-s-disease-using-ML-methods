import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ==========================================================
# Load 
# ==========================================================

dataset = pd.read_csv("TPC.csv") 
dataset
dataset.head()
print(dataset.Label.value_counts())
Y = dataset.Label
X = dataset.drop('Label', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("\nTrain shape:", X_train.shape, '\ty_train', Y_train.shape)
print("Test shape:", X_test.shape, '\ty_test', Y_test.shape)

df = pd.DataFrame(data=X_train)
df

from sklearn.preprocessing import LabelEncoder
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

from sklearn.preprocessing import MinMaxScaler as Scaler
scaler = Scaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

k = 5
kf = KFold(n_splits=k, random_state=None)
accuracies = cross_val_score(classifier, X_train, Y_train, cv=kf)

print("Avg accuracy: {}".format(accuracies.mean()))
accuracies.std()

def calculate_sensitivity_specificity(Y_test, Y_pred):
    # Note: More parameters are defined than necessary.
    # This would allow return of other measures other than sensitivity and specificity

    # Get true/false for whether a breach actually occurred
    actual_pos = Y_test == 1
    actual_neg = Y_test == 0

    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (Y_pred == 1) & (actual_pos)
    false_pos = (Y_pred == 1) & (actual_neg)
    true_neg = (Y_pred == 0) & (actual_neg)
    false_neg = (Y_pred == 0) & (actual_pos)

    # Calculate accuracy
    accuracy = np.mean(Y_pred == Y_test)

    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / np.sum(actual_pos)
    specificity = np.sum(true_neg) / np.sum(actual_neg)

    return sensitivity, specificity, accuracy

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(Y_test, Y_pred)
print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("Confusion Matrix: \n")
print(confusion_matrix(Y_test, Y_pred))
print("\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_pred))
print("\n")
print("Accuracy Score: \n")
print(accuracy_score(Y_test, Y_pred))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train, Y_train)
Y_predR = RF.predict(X_test)

k = 5
kf = KFold(n_splits=k, random_state=None)
accuracies = cross_val_score(classifier, X_train, Y_train, cv=kf)

print("Avg accuracy: {}".format(accuracies.mean()))
accuracies.std()

def calculate_sensitivity_specificity(Y_test, Y_predR):
    # Note: More parameters are defined than necessary.
    # This would allow return of other measures other than sensitivity and specificity

    # Get true/false for whether a breach actually occurred
    actual_pos = Y_test == 1
    actual_neg = Y_test == 0

    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (Y_predR == 1) & (actual_pos)
    false_pos = (Y_predR == 1) & (actual_neg)
    true_neg = (Y_predR == 0) & (actual_neg)
    false_neg = (Y_predR == 0) & (actual_pos)

    # Calculate accuracy
    accuracy = np.mean(Y_predR == Y_test)

    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / np.sum(actual_pos)
    specificity = np.sum(true_neg) / np.sum(actual_neg)

    return sensitivity, specificity, accuracy

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(Y_test, Y_predR)
print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix: \n")
print(confusion_matrix(Y_test, Y_predR))
print("\n")

from sklearn.metrics import classification_report
print("Classification Report: \n")
print(classification_report(Y_test, Y_predR))
print("\n")
print("Accuracy Score: \n")
print(accuracy_score(Y_test, Y_predR))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(90,90,90))
mlp.fit(X_train, Y_train)
Y_predM = mlp.predict(X_test)

k = 5
kf = KFold(n_splits=k, random_state=None)
accuracies = cross_val_score(classifier, X_train, Y_train, cv=kf)

print("Avg accuracy: {}".format(accuracies.mean()))

def calculate_sensitivity_specificity(Y_test, Y_predM):
    # Note: More parameters are defined than necessary.
    # This would allow return of other measures other than sensitivity and specificity

    # Get true/false for whether a breach actually occurred
    actual_pos = Y_test == 1
    actual_neg = Y_test == 0

    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (Y_predM == 1) & (actual_pos)
    false_pos = (Y_predM == 1) & (actual_neg)
    true_neg = (Y_predM == 0) & (actual_neg)
    false_neg = (Y_predM == 0) & (actual_pos)

    # Calculate accuracy
    accuracy = np.mean(Y_predM == Y_test)

    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / np.sum(actual_pos)
    specificity = np.sum(true_neg) / np.sum(actual_neg)

    return sensitivity, specificity, accuracy

sensitivity, specificity, accuracy = calculate_sensitivity_specificity(Y_test, Y_predM)
print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix: \n")
print(confusion_matrix(Y_test, Y_predM))
print("\n")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)

fpr1, tpr1, thresholds = roc_curve(Y_test, Y_predM)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, thresholds = roc_curve(Y_test, Y_predR)
roc_auc2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (SVM) %0.2f' % roc_auc)
plt.plot(fpr1, tpr1, color='red', lw=1, label='ROC curve (RF) %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, color='green', lw=1, label='ROC curve (ANN) %0.2f' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
