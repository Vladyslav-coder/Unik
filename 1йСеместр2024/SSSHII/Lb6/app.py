import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

num_words = 10000  
max_review_length = 500

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=num_words)

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 50
model = Sequential([
    Embedding(num_words, embedding_vector_length, input_length=max_review_length),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True)),  
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(46, activation='softmax')
])

optimizer = Adam(learning_rate=0.0005)  
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

train_history = model.fit(
    X_train, y_train, 
    epochs=20,  
    batch_size=16,  
    validation_split=0.2, 
    verbose=2
)

scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Точність: {scores[1] * 100:.2f}%")

def show_train_history(train_history, train_metric, val_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[val_metric])
    plt.title('Тренування')
    plt.ylabel(train_metric)
    plt.xlabel('Епоха')
    plt.legend(['Навчання', 'Валідація'], loc='upper left')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')
