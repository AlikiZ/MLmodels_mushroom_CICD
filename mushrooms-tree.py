import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlflow import log_metric, log_metrics, log_param, log_artifacts, set_experiment, start_run, end_run
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler # Für LogisticRegression
from sklearn.linear_model import LogisticRegression # Für LogisticRegression
from sklearn.naive_bayes import GaussianNB # Für NaiveBayes
from timeit import timeit, default_timer

set_experiment(experiment_name='Mushrooms')
#Daten werden eingelesen# #Ausprägungen sind noch Buchstaben in 23 Columns# # Ausprägungen sind 1en und 0en in 119 Columns#
df = pd.read_csv("./mushrooms.csv")
df = pd.get_dummies(df)

X = df.drop(["class_e","class_p"],axis=1).values
y = df["class_e"].values

# Split in Train- und Test-Daten
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.25)

############# Fit a model: DecisionTree ######################

depth = 0
for depth in range(1,9):
    start= default_timer()
    criterion = 'entropy'
    model_dt = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
    model_dt.fit(X_train,y_train)

    acc_dt = model_dt.score(X_test,y_test)
    stop = default_timer()
    print(acc_dt)
    log_metric("Accuracy Decision Tree - depth", acc_dt, depth)
    log_metric("Accuracy Decision Tree - runtime", stop - start, depth)

log_param("Depth of Decision Tree", depth)
log_param("Used Criterion for Decision Tree", criterion)

if not os.path.exists("outputs"):
        os.makedirs("outputs")
with open("outputs/metrics.txt", "a") as f:
        f.write("Accuracy_dt: " + str(acc_dt) + "\n")

# log_artifacts("outputs") ## Schreibt irgendwie auf Alikis PC

############# Fit a model: LogisticRegression ######################

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)
acc_lr = model_lr.score(X_test_scaled,y_test)
print(acc_lr)

with open("outputs/metrics.txt", "a") as f:
        f.write("Accuracy_lr: " + str(acc_lr) + "\n")


############# Fit a model: NaiveBayes ######################


model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

acc_nb= model_nb.score(X_test, y_test)
print(acc_nb)

log_metrics({
    #"Accuracy of DecisionTree": acc_dt,
    "Accuracy of Logistic Regression": acc_lr,
    "Accuracy of Naive Bayes": acc_nb
})

with open("outputs/metrics.txt", "a") as f:
        f.write("Accuracy_nb: " + str(acc_nb) + "\n")



# Plot confusion matrix and tree
disp = plot_confusion_matrix(model_dt, X_test, y_test, normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

plt.figure(dpi=100)
plot_tree(model_dt,fontsize=6,
          feature_names=df.columns,
          proportion=True,
          filled=True)

#plt.show()
# TO DO: file for cicd
