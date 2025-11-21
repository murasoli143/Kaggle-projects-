# Kaggle-projects-
Student performance prediction using machine learning 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")

# Data Cleaning
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Drop useless columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train.drop(columns=drop_cols, inplace=True)
test_saved = test.copy()
test.drop(columns=drop_cols, inplace=True)

# Encode categorical fields
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Sex'] = le.transform(test['Sex'])
test['Embarked'] = le.transform(test['Embarked'])

# Train-Test Split
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Final Prediction for submission
final_pred = model.predict(test)
submission = pd.DataFrame({
    'PassengerId': test_saved['PassengerId'],
    'Survived': final_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")
