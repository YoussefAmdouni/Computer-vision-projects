# This script utilizes 25,000 images downloaded from Kaggle and is inspired by a tutorial 
# from Kaggle's official documentation. Our GAN model can generate human face images with 
# basic facial features such as the face, nose, eyes, and mouth. However, we still encounter 
# noisy backgrounds and some imperfections in the facial images. To address this, we can 
# experiment with more complex architectures and incorporate additional images into the training process.



import keras
import tensorflow as tf

from tensorflow.keras import layers
import matplotlib.pyplot as plt

import numpy as np


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(28)
np.random.seed(28)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Conv2D(256, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",)
# discriminator.summary()



latent_dim = 1024

generator = keras.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(8 * 8 * 256, use_bias=False),
    layers.LeakyReLU(),
    layers.Reshape((8, 8, 256)),
    layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')
], name="generator")
# generator.summary()


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(28)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)

        generated_images = self.generator(random_latent_vectors)
        
        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)

        misleading_labels = tf.zeros((batch_size, 1))


        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result(),}



class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=1024):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(42)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))



mainDataPath = "Images/"  

image_scale = 128  
image_channels = 3  
images_color_mode = "rgb"  
image_shape = (image_scale, image_scale, image_channels)  

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_data_generator = ImageDataGenerator(rescale=1. / 255)
training_generator = training_data_generator.flow_from_directory(
    mainDataPath, 
    color_mode=images_color_mode, 
    target_size=(image_scale, image_scale),
    batch_size=28, 
    class_mode=None, 
    subset='training',
    classes=[''])  


epochs = 100

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(training_generator, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)])


random_latent_vectors = tf.random.normal(shape=(4, 1024))

generated_images = generator(random_latent_vectors)
generated_images *= 255
generated_images = generated_images.numpy()

plt.figure(figsize=(10, 2))
for i in range(generated_images.shape[0]):
    plt.subplot(1, generated_images.shape[0], i+1)
    plt.imshow(keras.utils.array_to_img(generated_images[i]))
    plt.axis('off')
plt.show()