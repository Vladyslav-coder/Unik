import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

train_file_path = 'C:\\Users\\baras\\\OneDrive\\Рабочий стол\\Unik\\SSSHII\\Lb7\\TestData.csv'
test_file_path = 'C:\\Users\\baras\\OneDrive\\Рабочий стол\\Unik\\SSSHII\\Lb7\\TrainData.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

num_classes = max(train_df['label'].max(), test_df['label'].max()) + 1

X_train = train_df.iloc[:, 1:].values
y_train = train_df['label'].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df['label'].values

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

input_shape = (28, 28, 1)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точність: {test_accuracy * 100:.2f}%")

model.save('C:\\Users\\baras\\OneDrive\\Рабочий стол\\Unik\\SSSHII\\Lb7\\sign_language_model.h5')
