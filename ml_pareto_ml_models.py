from ucimlrepo import fetch_ucirepo
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


import warnings
#warnings.filterwarnings("ignore")

# fetch dataset
#https://archive.ics.uci.edu/dataset/186/wine+quality
wine_quality = fetch_ucirepo(id=186) #vllt ist es keine gute Idee Rot- und Weißweine zu vermischen ;)

# data (as pandas dataframes)
X = wine_quality.data.features
print(X.describe())
y = wine_quality.data.targets
print(y.describe())

# Create different classifiers.
#classifiers_slow = {
#    "SVC (ovo)": svm.SVC(decision_function_shape='ovo'),
#    "MLP Classifier 15": MLPClassifier(max_iter=1000, hidden_layer_sizes=(15,)),
#}
classifiers = {
 "RFC (10)":  RandomForestClassifier(n_estimators=10),
 "RFC (20)":  RandomForestClassifier(n_estimators=20),
 "DTC (4)":   DecisionTreeClassifier(max_depth=4),
 "DTC (None)":DecisionTreeClassifier(max_depth=None),
 "KNC (10)":  KNeighborsClassifier(n_neighbors=10),
 "KNC (7)":   KNeighborsClassifier(n_neighbors=7),
 "KNC (5)":   KNeighborsClassifier(n_neighbors=5),
 "GNB":       GaussianNB(),
 "NearestCentroid (21)": NearestCentroid(shrink_threshold=21),
}

n_repetitions = 100 # we calculate mean value of all accuracies
calculated_values = []

for classifier_idx, (name, classifier) in enumerate(classifiers.items()):
    start_time = time.time()
    accuracies = []
    for iRep in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        curr_accuracy = accuracy_score(y_test.values.ravel(), classifier.fit(X_train, y_train.values.ravel()).predict(X_test))
        accuracies.append(curr_accuracy * 100.)
    end_time = time.time()
    accuracy = np.mean(accuracies)
    elapsed_time = 1000*(end_time - start_time) / n_repetitions
    print("Accuracy (test) for %s: %.1f%%, duration: %.2fms" % (name, accuracy, elapsed_time ))
    calculated_values.append((name, accuracy, elapsed_time, classifier_idx))

column_names = ['Model', 'Accuracy', 'Time', 'Idx']

df = pd.DataFrame(calculated_values, columns=column_names)

fig, (ax1) = plt.subplots(1,1)

plt.scatter(df['Time'], df['Accuracy'], c = df['Idx'], s=200, cmap='viridis')
plt.xlabel('Time [ms]')
plt.ylabel('Accuracy [%]')
plt.title('Accuracy vs Time')
plt.colorbar(label='Index')
fig.set_size_inches(15,10)
plt.show()


#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py