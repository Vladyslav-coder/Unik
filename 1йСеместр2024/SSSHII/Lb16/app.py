import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5  
train_images = np.expand_dims(train_images, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 20
NOISE_DIM = 100

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential([
        Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        LeakyReLU(),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Flatten(),
        Dense(1, activation='linear')  
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

mse_loss = tf.keras.losses.MeanSquaredError()
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

def generator_loss(fake_output):
    return mse_loss(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = mse_loss(tf.ones_like(real_output), real_output)
    fake_loss = mse_loss(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    gen_losses, disc_losses = [], []

    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss

        gen_losses.append(epoch_gen_loss / len(dataset))
        disc_losses.append(epoch_disc_loss / len(dataset))
        print(f'Epoch {epoch+1}, Gen Loss: {gen_losses[-1]}, Disc Loss: {disc_losses[-1]}')

    return gen_losses, disc_losses

gen_losses, disc_losses = train(train_dataset, EPOCHS)

plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label="Генератор")
plt.plot(disc_losses, label="Дискримінатор")
plt.xlabel("Епоха")
plt.ylabel("Втрата")
plt.legend()
plt.show()

def generate_images_for_date(date_str):
    noise = tf.random.normal([len(date_str), NOISE_DIM])
    generated_images = generator(noise, training=False)

    plt.figure(figsize=(15, 5))
    for i, char in enumerate(date_str):
        plt.subplot(1, len(date_str), i + 1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.title(char)

    plt.show()

generate_images_for_date("19.04.2003")
