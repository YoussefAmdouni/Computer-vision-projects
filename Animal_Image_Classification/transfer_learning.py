from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Activation, Dropout, Flatten, Dense

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

mainDataPath = "donnees/"
trainPath = mainDataPath + "entrainement"
validationPath = mainDataPath + "entrainement"
testPath = mainDataPath + "test"


training_batch_size = 32  
validation_batch_size = 32  
 
image_scale = 120 
image_channels = 3  
images_color_mode = "rgb"   
image_shape = (image_scale, image_scale, image_channels) 


input_layer = Input(shape=image_shape)

# ************************************************
#                Transfer Learning
# ************************************************
 
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Download Inception model from keras 
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (120, 120, 3), 
                                include_top = False, ) # We don't need the model last layer we will use a custom one
                                # weights = None) # we set weight to None if we want to train the model from scratch 
                                # otherwise it will use imagenet weights

pre_trained_model.load_weights(local_weights_file)
# We can freeze some weights "make them stable during training"
# for layer in pre_trained_model.layers:
#   layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
# we add 3 dense layer the last one contain only 6 neurone (#classes)
x = Flatten()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)   
x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)                
x = Dense(6, activation='softmax')(x)          
optimizer = SGD()

pretrained_model = Model(pre_trained_model.input, x) 

pretrained_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

pretrained_modelcheckpoint = ModelCheckpoint(filepath="pretrained.hdf5",
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')


training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.05)


training_generator = training_data_generator.flow_from_directory(
    trainPath, 
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=32, 
    shuffle=True, 
    subset='training')

validation_generator = training_data_generator.flow_from_directory(
    validationPath, 
    color_mode=images_color_mode, 
    target_size=(image_scale, image_scale),  
    batch_size=32,  
    shuffle=True, 
    subset='validation')


history = pretrained_model.fit(
            training_generator,
            batch_size = 32,
            validation_data = validation_generator,
            callbacks=[pretrained_modelcheckpoint],
            epochs = 10,
            verbose = 1)


#### Evaluation on test data ####
number_images = 6000
number_images_class_0 = 1000
number_images_class_1 = 1000
number_images_class_2 = 1000
number_images_class_3 = 1000
number_images_class_4 = 1000
number_images_class_5 = 1000

test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,
    target_size=(image_scale, image_scale), 
    shuffle=False,
    batch_size=1,
    color_mode=images_color_mode)


y_true = np.array([0] * number_images_class_0 + 
                  [1] * number_images_class_1 + 
                  [2] * number_images_class_0 + 
                  [3] * number_images_class_1 +
                  [4] * number_images_class_0 + 
                  [5] * number_images_class_1 )

test_eval = pretrained_model.evaluate_generator(test_itr, verbose=1)

print('Test loss :', test_eval[0]) # 0.1291
print('Test accuracy:', test_eval[1]) # 0.9625