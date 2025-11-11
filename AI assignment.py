import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Load the dataset
df = pd.read_csv('training_data.csv')

# Convert labels to numbers for classification
df['Risk_bin'] = df['Risk'].map({'bad': 1, 'good': 0})

#Manual Rule-Based System (simple student example)
def rule_based(row):
    if row['Credit amount'] > 10000:
        return 1  #high risk
    if row['Age'] < 25 and row['Credit amount'] > 3000:
        return 1  #high risk
    return 0  #low risk

df['rule_pred'] = df.apply(rule_based, axis=1)
rule_accuracy = accuracy_score(df['Risk_bin'], df['rule_pred'])
rule_conf = confusion_matrix(df['Risk_bin'], df['rule_pred'])

#Decision Tree (basic, numeric features only)
features = ['Age', 'Job', 'Credit amount', 'Duration']
X = df[features].fillna(0)
y = df['Risk_bin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_conf = confusion_matrix(y_test, dt_pred)

print('Rule-Based System:')
print(f'Accuracy: {rule_accuracy:.3f}')
print('Confusion Matrix:')
print(rule_conf)
print()
print('Decision Tree Classifier:')
print(f'Accuracy: {dt_accuracy:.3f}')
print('Confusion Matrix:')
print(dt_conf)
