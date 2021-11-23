import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Daten werden eingelesen#
df = pd.read_csv("./mushrooms.csv") #Ausprägungen sind noch Buchstaben in 23 Columns
df = pd.get_dummies(df) # Ausprägungen sind 1en und 0en in 119 Columns

X = df.drop(["class_e","class_p"],axis=1).values # Trennung von X und Y, genommen werden X-Daten
y = df["class_e"].values # Trennung von X und Y, genommen werden Y-Daten

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.25) # Split in Train- und Test-Daten

model = DecisionTreeClassifier(criterion='entropy',max_depth=5)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))



plt.figure(dpi=100)
plot_tree(model,fontsize=6,
          feature_names=df.columns,
          proportion=True,
          filled=True)

plt.show()