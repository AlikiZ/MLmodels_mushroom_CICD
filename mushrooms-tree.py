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

set_experiment(experiment_name='ToxicMushrooms')
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
    log_metric("Decision Tree - accuracy vs depth", acc_dt, depth)
    log_metric("Decision Tree - runtime vs depth", stop - start, depth)

    #log_param("Depth of Decision Tree", depth)
    #log_param("Used Criterion for Decision Tree", criterion)

    if not os.path.exists("outputs"):
            os.makedirs("outputs")
    if depth == 1:
        with open("outputs/metrics_decision_tree.txt", "w") as f:
                f.write("Model: Decision Tree \n\n")
                f.write("Depth Accuracy Runtime \n")
                f.write( str(depth) + "    " + str(acc_dt)[0:5] + "  " + str(stop - start)[0:5] + "\n")
    else:
        with open("outputs/metrics_decision_tree.txt", "a") as f:
                f.write(str(depth) + "  " + str(acc_dt)[0:5] + "    " + str(stop - start)[0:5] + "\n")


#log_artifacts("outputs") ## Schreibt irgendwie auf Alikis PC

############# Fit a model: LogisticRegression ######################

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

solver = ['newton-cg','lbfgs','liblinear','sag','saga']
for entry in range(0,len(solver)):
    start = default_timer()
    model_lr = LogisticRegression(solver= solver[entry])
    model_lr.fit(X_train_scaled, y_train)
    acc_lr = model_lr.score(X_test_scaled,y_test)
    print(acc_lr)
    stop = default_timer()
    log_metric("Logistic Regression - accuracy vs solver", acc_dt, entry)
    log_metric("Logistic Regression - runtime vs solver", stop - start, entry)

    if entry == 0:
        with open("outputs/metrics_logistic_regression.txt", "w") as f:
            f.write("Model: Logistic Regression \n\n")
            f.write("  Solver Accuracy Runtime \n")
            f.write(str(entry) + " " + str(solver[entry]) + "  " + str(acc_lr) + " " + str(stop - start)[0:5] + "\n")
    else:
        with open("outputs/metrics_logistic_regression.txt", "a") as f:
            f.write(str(entry) + " " + str(solver[entry]) + "  " + str(acc_lr) + " " + str(stop - start)[0:5] + "\n")


############# Fit a model: NaiveBayes ######################

smoothing = [1e-04,1e-05,1e-06,1e-07,1e-08,1e-09,1e-10,1e-11]

for entry in range(0,len(smoothing)):
    start = default_timer()
    model_nb = GaussianNB(var_smoothing=smoothing[entry])
    model_nb.fit(X_train, y_train)

    acc_nb= model_nb.score(X_test, y_test)
    print(acc_nb)
    stop = default_timer()
    log_metric("Gaussian Naive Bayes - accuracy vs smoothing", acc_nb, entry)
    log_metric("Gaussian Naive Bayes - runtime vs smoothing", stop - start, entry)

    if entry == 0:
        with open("outputs/metrics_naive_bayes.txt", "w") as f:
            f.write("Model: Gaussian Naive Bayes \n\n")
            f.write("  Smoothing Accuracy Runtime \n")
            f.write(str(entry) + " " + str(smoothing[entry]) + "  " + str(acc_nb)[0:5] + " " + str(stop - start)[0:5] + "\n")
    else:
        with open("outputs/metrics_naive_bayes.txt", "a") as f:
            f.write(str(entry) + " " + str(smoothing[entry]) + "  " + str(acc_nb)[0:5] + " " + str(stop - start)[0:5] + "\n")



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

log_artifacts("outputs") ## Schreibt irgendwie auf Alikis PC
