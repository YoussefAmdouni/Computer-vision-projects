# This script utilizes 25,000 images downloaded from Kaggle and is inspired by a tutorial 
# from Kaggle's official documentation. Our VAE model can generate human face images with 
# basic facial features such as the face, nose, eyes, and mouth. However, we still encounter 
# noisy backgrounds and some imperfections in the facial images. To address this, we can 
# experiment with more complex architectures and incorporate additional images into the training process.

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
    

latent_dim = 1024

encoder_inputs = keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(16, 3, strides=2, padding="same", activation='relu')(encoder_inputs)
x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(2048, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_var  = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_var, z], name="encoder")
# encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss, self.reconstruction_loss, self.kl_loss]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
       
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.kl_loss.update_state(kl_loss)
        
        return {"loss": self.total_loss.result(),
                "reconstruction_loss": self.reconstruction_loss.result(),
                "kl_loss": self.kl_loss.result(),}
    

mainDataPath = "Images/"

image_scale = 128  
image_channels = 3  
images_color_mode = "rgb"  
image_shape = (image_scale, image_scale, image_channels)  

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

training_data_generator = ImageDataGenerator(rescale=1. / 255)

training_generator = training_data_generator.flow_from_directory(
    mainDataPath, 
    color_mode=images_color_mode, 
    target_size=(image_scale, image_scale),
    batch_size=28, 
    class_mode=None, 
    subset='training',
    classes=[''])  


vae = VAE(encoder, decoder)
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
vae.compile(optimizer=optimizer)

vae.fit(training_generator, epochs=100)


from PIL import Image
import matplotlib.pyplot as plt


latent_samples = np.random.normal(size=(4, latent_dim))  
generated_images = decoder.predict(latent_samples)

plt.figure(figsize=(10, 2))
for i in range(generated_images.shape[0]):
    plt.subplot(1, generated_images.shape[0], i+1)
    plt.imshow(generated_images[i])
    plt.axis('off')
    plt.savefig('eval_results/vae_images.png')
plt.show()