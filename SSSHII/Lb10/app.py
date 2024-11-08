from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import numpy as np

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=3, random_state=1)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

model.fit(X, y)
row = [[5.1, 3.5, 1.4, 0.2]]  
yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])
