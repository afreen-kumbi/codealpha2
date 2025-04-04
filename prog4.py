# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Create DataFrame with original data
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[y]
# Display first 5 rows of original data
print("First 5 rows of original dataset:")
print(df.head())
print("\n" + "="*80 + "\n")
# Encode species labels
le = LabelEncoder()
df['encoded_species'] = le.fit_transform(df['species'])
# Display first 5 rows after encoding
print("First 5 rows after encoding:")
print(df[['sepal length (cm)', 'sepal width (cm)',
 'petal length (cm)', 'petal width (cm)',
 'species', 'encoded_species']].head())
print("\n" + "="*80 + "\n")
# Split data into features and target
X = df.drop(['species', 'encoded_species'], axis=1)
y = df['encoded_species']
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
)
# Create and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Make predictions
y_pred = knn.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate classification report
clf_report = classification_report(y_test, y_pred,
 target_names=iris.target_names)
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(clf_report)
print("\nConfusion Matrix:")
print(conf_matrix)
