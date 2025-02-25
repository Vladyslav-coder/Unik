import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:\\Users\\baras\\OneDrive\\Рабочий стол\\Unik\\SSSHII\\LB15\\orange_quality.csv')

label_encoder = LabelEncoder()
data['Color'] = label_encoder.fit_transform(data['Color'])
data['Variety'] = label_encoder.fit_transform(data['Variety'])
data['Blemishes (Y/N)'] = label_encoder.fit_transform(data['Blemishes (Y/N)'])

X = data.drop('Quality (1-5)', axis=1)
y = data['Quality (1-5)'].round().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

mlp_predictions = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"Точність нейронної мережі: {mlp_accuracy:.2f}")

gb = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1.0],
    'max_depth': [3, 5]
}

grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Найкращі параметри: {best_params}")
print(f"Точність з найкращими параметрами: {best_score:.2f}")