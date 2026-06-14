import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Marks': [85, 78, 92, 67, 74,
              88, 45, 55, 95, 60],

    'Attendance': [90, 88, 95, 80, 85,
                   92, 60, 70, 98, 75],

    'Result': ['Pass', 'Pass', 'Pass', 'Pass',
               'Pass', 'Pass', 'Fail',
               'Fail', 'Pass', 'Fail']
}

df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

df['Result'] = df['Result'].map({
    'Fail': 0,
    'Pass': 1
})

X = df[['Marks', 'Attendance']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nProject Completed Successfully!")


seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week


import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)

