import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.font_manager
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test


# Generate train data
with open('normal_data.json', 'r') as f:
    data = json.load(f)

features = np.array(data["features"])
labels = np.array(data["labels"])
clean = features[labels == 0]

outlier_sets = {
    'poop2_clog.json': 'gold',
    'food_clog.json': 'tomato'
}
outlier_data = {}
for filename, color in outlier_sets.items():
    with open(filename, 'r') as f:
        temp = json.load(f)
    outlier_data[filename] = (np.array(temp["features"]), color)

# leads to 70,15,15 split
X_train, X_temp = train_test_split(clean, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)



# defines model hyper params
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1) # defines classifers params
#actually trains model
clf.fit(X_train)

# returns 1 for normal data and -1 for abnormal data
y_pred_train = clf.predict(X_train) # runs train set through model
y_pred_test = clf.predict(X_test) # runs test set(40 dots) through model
outlier_preds = {}
for filename, (X_out, color) in outlier_data.items():
    y_pred = clf.predict(X_out)
    outlier_preds[filename] = y_pred

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

_, ax = plt.subplots()
pca = PCA(n_components=2)
all_data = np.vstack([X_train, X_test] + [X for X, _ in outlier_data.values()])
pca.fit(all_data)

s = 40
b1 = ax.scatter(*pca.transform(X_train).T, c="white", s=s, edgecolors="k")
b2 = ax.scatter(*pca.transform(X_test).T, c="blueviolet", s=s, edgecolors="k")

outlier_handles = [b1, b2]
outlier_labels = ["training observations", "new regular observations"]
for filename, (X_out, color) in outlier_data.items():
    y_pred = outlier_preds[filename]
    n_err = y_pred[y_pred == 1].size
    h = ax.scatter(*pca.transform(X_out).T, c=color, s=s, edgecolors="k")
    outlier_handles.append(h)
    outlier_labels.append(f"{filename} ({n_err}/{len(X_out)} missed)")
    print(f"errors novel abnormal ({filename}): {n_err}/{len(X_out)}")

plt.legend(outlier_handles, outlier_labels, loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
ax.set(
    xlabel=f"error train: {n_error_train}/{len(X_train)} ; errors novel regular: {n_error_test}/{len(X_test)}",
    title="Novelty Detection",
)
print(f"error train: {n_error_train}/{len(X_train)} ; errors novel regular: {n_error_test}/{len(X_test)}")
        
plt.show()