import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

==========================================

1. LOAD DATASET

==========================================

df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

==========================================

2. EXPLORE DATASET

==========================================

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSpecies distribution:")
print(df["Species"].value_counts())

==========================================

3. SELECT FEATURES AND TARGET

==========================================

X = df[
[
"SepalLengthCm",
"SepalWidthCm",
"PetalLengthCm",
"PetalWidthCm"
]
]

y = df["Species"]

==========================================

4. SPLIT DATA INTO TRAINING AND TESTING

==========================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

==========================================

5. CREATE MACHINE LEARNING MODEL

==========================================

model = LogisticRegression(max_iter=200)

==========================================

6. TRAIN THE MODEL

==========================================

model.fit(X_train, y_train)

==========================================

7. MAKE PREDICTIONS

==========================================

y_pred = model.predict(X_test)

==========================================

8. CHECK ACCURACY

==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

==========================================

9. CLASSIFICATION REPORT

==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

==========================================

10. CONFUSION MATRIX

==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=model.classes_,
yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

==========================================

11. DATA VISUALIZATION

==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
data=df,
x="PetalLengthCm",
y="PetalWidthCm",
hue="Species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.savefig("petal_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

==========================================

12. PREDICT A NEW FLOWER

==========================================

new_flower = pd.DataFrame(
[[5.1, 3.5, 1.4, 0.2]],
columns=[
"SepalLengthCm",
"SepalWidthCm",
"PetalLengthCm",
"PetalWidthCm"
]
)

prediction = model.predict(new_flower)

print("\nPredicted species:", prediction[0])

is this correct