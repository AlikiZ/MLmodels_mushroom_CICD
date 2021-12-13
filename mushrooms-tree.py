import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlflow import log_metric, log_param, log_artifacts
import os
from sklearn.metrics import plot_confusion_matrix

#Daten werden eingelesen# #Ausprägungen sind noch Buchstaben in 23 Columns# # Ausprägungen sind 1en und 0en in 119 Columns#
df = pd.read_csv("./mushrooms.csv")
df = pd.get_dummies(df)

X = df.drop(["class_e","class_p"],axis=1).values
y = df["class_e"].values

# Split in Train- und Test-Daten
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.25)

# Fit a model
depth = 8
criterion = 'entropy'
model = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print(acc)

# incorporate mlflow for tracking
log_param("depth", depth)
log_param("criterion", criterion)

log_metric("accuracy", acc)

if not os.path.exists("outputs"):
        os.makedirs("outputs")
with open("outputs/metrics.txt", "w") as f:
        f.write("Accuracy: " + str(acc) + "\n")

log_artifacts("outputs")

# Plot confusion matrix and tree
disp = plot_confusion_matrix(model, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

plt.figure(dpi=100)
plot_tree(model,fontsize=6,
          feature_names=df.columns,
          proportion=True,
          filled=True)

plt.show()

# TO DO: file for cicd