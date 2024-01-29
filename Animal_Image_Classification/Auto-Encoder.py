from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, Flatten

from tensorflow.keras.callbacks import ModelCheckpoint


import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

tf.random.set_seed(1996)
np.random.seed(1996)

mainDataPath = "data/"
trainPath = mainDataPath + "entrainement"
validationPath = mainDataPath + "entrainement"
model_path = "AutoEncoder.hdf5"


image_scale = 128
image_channels = 3
images_color_mode = "rgb"
image_shape = (image_scale, image_scale, image_channels)


def encoder(inputs):
    conv_1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv1')(inputs)
    conv_1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv2')(conv_1)
    max_pool_1 = MaxPooling2D(pool_size = (2, 2), name = 'encoder_mp1')(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv3')(max_pool_1)
    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv4')(conv_2)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), name = 'encoder_mp2')(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv5')(max_pool_2)
    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv6')(conv_3)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), name = 'encoder_mp3')(conv_3)

    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv7')(max_pool_3)
    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name = 'encoder_conv8')(conv_4)
    encoded = MaxPooling2D(pool_size=(2, 2), name = 'encoder_mp4')(conv_4)
    return encoded


def bottle_neck(inputs):
    bottle_neck = inputs
    encoder_visualization = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='sigmoid', padding='same')(bottle_neck)
    flatten_output = Flatten(name = 'bottle_neck')(inputs)
    return bottle_neck, encoder_visualization, flatten_output


def decoder(inputs):
    conv_1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv1')(inputs)
    conv_1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv2')(conv_1)
    up_sample_1 = UpSampling2D(size=(2, 2), name = 'decoder_mp1')(conv_1)

    conv_2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv3')(up_sample_1)
    conv_2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv4')(conv_2)
    up_sample_2 = UpSampling2D(size=(2, 2), name = 'decoder_mp2')(conv_2)

    conv_3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv5')(up_sample_2)
    conv_3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv6')(conv_3)
    up_sample_3 = UpSampling2D(size=(2, 2), name = 'decoder_mp3')(conv_3)

    conv_4 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv7')(up_sample_3)
    conv_4 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', name = 'decoder_conv8')(conv_4)
    up_sample_4 = UpSampling2D(size=(2, 2), name = 'decoder_mp4')(conv_4)

    decoded = Conv2D(image_channels, (3, 3), activation='sigmoid', padding='same', name = 'decoder_conv9')(up_sample_4)
    return decoded


def convolutional_auto_encoder():
    inputs = Input(shape=image_shape)
    encoder_output = encoder(inputs)
    bottleneck_output, encoder_visualization, flatten_output = bottle_neck(encoder_output)
    decoder_output = decoder(bottleneck_output)

    model = Model(inputs =inputs, outputs=decoder_output)
    encoder_model = Model(inputs=inputs, outputs=flatten_output)
    return model, encoder_model, encoder_visualization

model, encoder_model, encoder_visualization = convolutional_auto_encoder()
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.05,)


training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode = images_color_mode,
    target_size = (image_scale, image_scale),
    batch_size = 64,
    class_mode ="input",
    subset='training')

validation_generator = training_data_generator.flow_from_directory(
    validationPath,
    color_mode =images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size = 64,
    class_mode ="input",
    subset='validation')


(x_train, _) = training_generator.next()
(x_val, _) = validation_generator.next()
x_train.shape
x_val.shape

modelcheckpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

autoencoder = model.fit(training_generator,
                       epochs=300,
                       verbose=1,
                       callbacks=[modelcheckpoint],
                       shuffle=False,
                       validation_data=validation_generator)

plt.imshow(x_train[50])

temp = model.predict((x_train[50].reshape(1,128,128,3)))
temp = temp.reshape(128,128,3)
plt.imshow(temp)

plt.plot(autoencoder.history['loss'])
plt.plot(autoencoder.history['val_loss'])
plt.title('model loss with data augmentation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.grid()
plt.show()

def test_gen(image_scale, images_color_mode):
  mainDataPath = "data/"

  datapath = mainDataPath + "test"

  number_images = 2000
  number_images_class_0 = 1000
  number_images_class_1 = 1000

  labels = np.array([0] * number_images_class_0 +
                    [1] * number_images_class_1 )

  data_generator = ImageDataGenerator(rescale=1. / 255)

  generator = data_generator.flow_from_directory(
      datapath,
      color_mode=images_color_mode,
      target_size=(image_scale, image_scale),
      batch_size= number_images,
      class_mode=None,
      shuffle=False)

  x = generator.next()
  return x, labels

x, labels = test_gen(128, 'rgb')
x.shape


from tensorflow.keras.models import load_model

autoencoder = load_model("ModelAutoEncode.hdf5")
autoencoder.summary()

encoded = Model(autoencoder.layers[0].input, autoencoder.layers[12].output)
last_layer = encoded.get_layer('encoder_mp4')
last_output = last_layer.output
last_output = Flatten()(last_output)
encoder = Model(autoencoder.input, last_output)
encoder.summary()

x_embed_test = encoder_model.predict(x)
x_embed_test = encoder.predict(x)
x_embed_test.shape



scaler = StandardScaler()
scaler.fit(x_embed_test)
x_embed_test_sc = scaler.transform(x_embed_test)

lr = LogisticRegression()
scores = cross_val_score(lr, x_embed_test, labels, cv=5,)
scores.mean()


def plot_tsne(data):
    tsne = TSNE(n_components=2)
    images_tsne = tsne.fit_transform(data)

    sns.scatterplot(x=images_tsne[:,0],
    y=images_tsne[:,1],
    hue = labels).set(title = "Le embedding en deux dimensions:TSNE")

plot_tsne(x_embed_test)


def show_images(x, model):
    elephant = random.randint(0,999)
    tigre = random.randint(1000, 1999)

    temp = x[[elephant, tigre]]
    pred = model.predict(temp)
    plt.figure(figsize=(16, 8))

    plt.subplot(2,2,1)
    plt.title('Original Image')
    plt.imshow(temp[0])
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.title('Original Image')
    plt.imshow(temp[1])
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.title('Predicted Image')
    plt.imshow(pred[0])
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.title('Predicted Image')
    plt.imshow(pred[1])
    plt.axis('off')
    plt.show()

show_images(x, autoencoder)