from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam, SGD

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
modelsPath = "Classification_Model.hdf5"


training_batch_size = 32  
validation_batch_size = 32  
 
image_scale = 120 
image_channels = 3  
images_color_mode = "rgb"   
image_shape = (image_scale, image_scale, image_channels) 


input_layer = Input(shape=image_shape)


def feature_extraction(input):
    x = Conv2D(32, (5, 5), padding='same')(input) 
    x = Activation("LeakyReLU")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (5, 5), padding='same')(x) 
    x = Activation("LeakyReLU")(x)
    x = MaxPooling2D((2, 2))(x)   

    x = Conv2D(128, (5, 5), padding='same')(x) 
    x = Activation("LeakyReLU")(x)
    x = MaxPooling2D((2, 2))(x)    
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation("LeakyReLU")(x)
    encoded = MaxPooling2D((2, 2))(x)  
    return encoded


def fully_connected(encoded):
    x = Flatten(input_shape=image_shape)(encoded)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = Activation("LeakyReLU")(x)
    x = Dense(512)(x)
    x = Dropout(0.25)(x)
    x = Activation("LeakyReLU")(x)
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


model = Model(input_layer, fully_connected(feature_extraction(input_layer)))
model.summary()


optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


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


print(training_generator.class_indices)
print(validation_generator.class_indices)

(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value


# =========== Training =============

modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

import time 
start = time.time()
classifier = model.fit(training_generator,
                       epochs=120, 
                       batch_size=32, 
                       validation_data=validation_generator, 
                       verbose=1, 
                       callbacks=[modelcheckpoint], 
                       shuffle=True)
print(time.time()-start)

# ======== Results ===========

print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()


print(classifier.history.keys())
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()


#### Evaluation on test data ####

from sklearn.metrics import confusion_matrix, roc_curve , auc, accuracy_score
from keras.models import load_model


trained_model = load_model("Classification_Model.hdf5")

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

test_eval = trained_model.evaluate_generator(test_itr, verbose=1)

print('Test loss :', test_eval[0]) # 0.2502
print('Test accuracy:', test_eval[1]) # 0.9225


# Plot model
from keras.utils.vis_utils import plot_model
plot_model(trained_model, to_file = 'model_plot.png',
           show_shapes = True,
           show_layer_names = True)


predicted_classes = trained_model.predict_generator(test_itr, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predictions = predicted_classes.argmax(axis=-1)


def func_pred(predicted_classes, y_true):
    correct = []
    incorrect = []
    for i in range(0, len(predicted_classes)):
        if predictions[i] == y_true[i]:
            correct.append(i)
        else:
            incorrect.append(i)
    print("%d  images well classified" % len(correct))
    print("%d images miss classified" % len(incorrect))

func_pred(predicted_classes, y_true)

# 5535  images well classified
# 465 images miss classified


# Confusion matrix
import itertools

def conf_mat(y_true, y_pred):
    ConfusionMatrix = confusion_matrix(y_true, y_pred) 
    fig = plt.figure(figsize=(12,6))
    plt.imshow(ConfusionMatrix, cmap =plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.ylabel('True label')

    plt.colorbar()

    target_names = ['elephant', 'girafe', 'leopard', 'rhino', 'tigre', 'zebre']
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)

    thresh = ConfusionMatrix.max() / 2.
    for i, j in itertools.product(range(ConfusionMatrix.shape[0]), range(ConfusionMatrix.shape[1])):
        plt.text(j, i, ConfusionMatrix[i, j],
                horizontalalignment="center",
                color="white" if ConfusionMatrix[i, j] > thresh else "black")
    plt.savefig('eval_results/Confusion_matrix.png')
    plt.show()

conf_mat(y_true, predictions)


# ROC Plot
from sklearn.preprocessing import label_binarize 


def roc_multi(y_true, y_pred):
    
    y_true = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
    y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5])

    mapper = {0:'elephant', 1:'girafe', 2:'leopard', 3:'rhino', 4:'tigre', 5:'zebre'}

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(6):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 6

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]), color="deeppink", linestyle=":", linewidth=2,)

    plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]), color="navy", linestyle=":", linewidth=2,)

    for i in range(6):
        plt.plot(fpr[i], tpr[i], label="Class {0} (area = {1:0.2f})".format(mapper[i], roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="best")
    plt.show()

roc_multi(y_true, predictions)



# Visualizing Intermediate Representations
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def intermediate_representation_sample(model, img_path):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = Model(inputs = model.input, outputs = successive_outputs)

    img = load_img(img_path, target_size=(120, 120))  

    img = img_to_array(img)                         
    img = img.reshape((1,) + img.shape)             
    img /= 255.0

    successive_feature_maps = visualization_model.predict(img)
    layer_names = [layer.name for layer in model.layers]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):    
        if len(feature_map.shape) == 4:       
            # n_features = feature_map.shape[-1]  
            size       = feature_map.shape[ 1]  

            display_grid = np.zeros((size, size * 5))
        
            for i in range(5):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = x 


            scale = 20/5
            plt.figure( figsize=(scale * 5, scale) )
            plt.axis('Off')
            plt.title ( layer_name )
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
            plt.savefig(f'eval_results/{layer_name}.png')

intermediate_representation_sample(trained_model, 'donnees/entrainement/elephant/0011.png')



# Extract 5 images of each category
import random 
import os
import matplotlib.image as mpimg
def get_images():
    all_img = []
    dirs = os.listdir('donnees/entrainement/')
    for dir in dirs:
        path = 'donnees/entrainement/' + str(dir) + '/'
        images_path = os.listdir(path)
        choice = random.choices(images_path, k=5)
        choice = [os.path.join(path, e) for e in choice]
        all_img = all_img + choice
    return all_img

def plot_images(paths):
    for i, img_path in enumerate(paths):
        ax = plt.subplot(6, 5, i + 1)
        ax.axis('Off')
        img = mpimg.imread(img_path, 0)
        plt.imshow(img)
    plt.savefig('eval_results/sample_images.png')
    plt.show()

plot_images(get_images())


# Show Some misclassification (randomly taken)
test_itr = test_data_generator.flow_from_directory(
    testPath,
    target_size=(image_scale, image_scale), 
    shuffle=False,
    batch_size=6000,
    color_mode=images_color_mode)

(x_test, y_test) = test_itr.next()

def plot_misclassification(x_test, y_true, predictions):
    mapper = {0:'elephant', 1:'girafe', 2:'leopard', 3:'rhino', 4:'tigre', 5:'zebre'}

    errors = (predictions - y_true != 0)
    errors_idx = np.where(errors)[0]
    choice_errors = random.choices(errors_idx, k=8)

    n = 0
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.1)
    for row in range(2):
        for col in range(4):
            ax[row,col].imshow(x_test[choice_errors[n]])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(mapper[predictions[choice_errors[n]]],
                mapper[y_true[choice_errors[n]]]), fontsize = 7)
            n += 1
            ax[row,col].axis('Off')
    plt.savefig('eval_results/miss_classification.png')
    plt.show()

plot_misclassification(x_test, y_true, predictions)

# show 5 misclassification for each class
def plot_five_miss(x_test, y_true, predictions):
    mapper = {0:'elephant', 1:'girafe', 2:'leopard', 3:'rhino', 4:'tigre', 5:'zebre'}

    errors = (predictions - y_true != 0)
    errors_idx = np.where(errors)[0]

    idx_elephant = random.choices([i for i in errors_idx if i<1000], k=5)
    idx_girafe  = random.choices([i for i in errors_idx if 1000<=i<2000], k=5)
    idx_leopard  = random.choices([i for i in errors_idx if 2000<=i<3000], k=5)
    idx_rhino = random.choices([i for i in errors_idx if 3000<=i<4000], k=5)
    idx_tigre  = random.choices([i for i in errors_idx if 4000<=i<5000], k=5)
    idx_zebre  = random.choices([i for i in errors_idx if i<6000], k=5)

    all_ind = idx_elephant + idx_girafe + idx_leopard + idx_rhino + idx_tigre + idx_zebre

    n = 0
    fig, ax = plt.subplots(6, 5, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.6)
    for row in range(6):
        for col in range(5):
            ax[row,col].imshow(x_test[all_ind[n]])
            ax[row,col].set_title("Predicted:{}\nTrue:{}".format(mapper[predictions[all_ind[n]]],
                mapper[y_true[all_ind[n]]]), fontsize = 6)
            n += 1
            ax[row,col].axis('Off')
    plt.show()
plot_five_miss(x_test, y_true, predictions)