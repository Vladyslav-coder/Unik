from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

X, y = make_blobs(n_samples=50000, centers=3, n_features=2, random_state=2)
y = to_categorical(y)

def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

k = 10
kfold = KFold(n_splits=k, shuffle=True, random_state=1)
cv_scores = []

for train_ix, test_ix in kfold.split(X):
    trainX, testX = X[train_ix], X[test_ix]
    trainy, testy = y[train_ix], y[test_ix]
    model = create_model()
    model.fit(trainX, trainy, epochs=5, verbose=0)
    _, acc = model.evaluate(testX, testy, verbose=0)
    print(f'Точність для складу: {acc:.3f}')
    cv_scores.append(acc)

print(f'Середня точність для k-кратної перехресної перевірки: {np.mean(cv_scores):.3f}')

testX, testy = X[100:], y[100:]
testy_enc = to_categorical(np.argmax(testy, axis=1))

print(f"Розмір testX: {len(testX)}")
print(f"Розмір testy_enc: {len(testy_enc)}")

min_samples = min(len(testX), len(testy_enc))
testX = testX[:min_samples]
testy_enc = testy_enc[:min_samples]

_, acc = model.evaluate(testX, testy_enc, verbose=0)
print(f'Точність на тестових даних: {acc:.3f}')