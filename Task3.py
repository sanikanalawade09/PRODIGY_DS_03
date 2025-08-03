# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the Bank Marketing dataset from your local CSV file
df = pd.read_csv("bank.csv")  # Replace with your actual file path

# 3. Quick data overview
print("Shape:", df.shape)
print(df.head())
print(df.info())

# 4. Print class distribution before encoding
print("\nTarget variable distribution (before encoding):")
print(df['y'].value_counts(normalize=True))

# 5. Encode categorical variables using Label Encoding
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 6. Define features and target variable
X = df.drop('y', axis=1)  # Features
y = df['y']               # Target: subscription (0 = no, 1 = yes)

# 7. Split data into train and test sets with stratification (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 9. Predict on test set
y_pred = clf.predict(X_test)

# 10. Evaluate model performance
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Decision Tree for Bank Marketing Prediction")
plt.show()

# 12. Feature Importance visualization
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance")
plt.show()
