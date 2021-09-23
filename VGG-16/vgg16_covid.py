import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sn

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from numpy.random import seed
seed(8) 

tf.random.set_seed(7) 

#%%

def init_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            #Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            #Memory growth must be set before GPUs have been initialized
            print(e)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#%%
init_gpus()
DATASET_PATH  = './data/all/train'
test_dir =  './data/all/test'
IMAGE_SIZE    = (150, 150)
data_list = ['normal', 'pneumonia', 'covid']
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 30
LEARNING_RATE =0.0001

#%%
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   featurewise_center = True,
                                   featurewise_std_normalization = True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.25,
                                   zoom_range=0.1,
                                   zca_whitening = True,
                                   channel_shift_range = 20,
                                   horizontal_flip = True ,
                                   vertical_flip = True ,
                                   validation_split = 0.2,
                                   fill_mode='constant')
                                   
#%%
train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  )
#%%
valid_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                 
                                                  )
#%%
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


conv_base.trainable = False


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', min_delta = 1e-6, patience = 5, verbose =1)


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['acc', f1_m,precision_m, recall_m])


#%%

print(len(train_batches))
print(len(valid_batches))

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

result=model.fit_generator(train_batches,
                        steps_per_epoch =STEP_SIZE_TRAIN,
                        validation_data = valid_batches,
                        validation_steps = STEP_SIZE_VALID,
                        epochs= NUM_EPOCHS,                        
                       verbose=2,  callbacks = [es])


# model.save('.models/Covid_Multi.h5')
#%%
def plot_acc_loss(result, epochs):
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(1,epochs), acc[1:], label='Train_acc')
    plt.plot(range(1,epochs), val_acc[1:], label='Test_acc')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(1,epochs), loss[1:], label='Train_loss')
    plt.plot(range(1,epochs), val_loss[1:], label='Test_loss')
    plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/acc_loss.png', dpi=300)
number_of_epochs = len(result.history['loss']) 
plot_acc_loss(result, number_of_epochs)


#%%
test_datagen = ImageDataGenerator(rescale=1. / 255)

eval_generator = test_datagen.flow_from_directory(
        test_dir,target_size=IMAGE_SIZE,
        batch_size=1,
        shuffle=True,
        seed=42,
        class_mode="categorical")
eval_generator.reset()
#%%
eval_generator.reset()  
loss, accuracy, f1_score, precision, recall = model.evaluate_generator(eval_generator,
                           steps = np.ceil(len(eval_generator) / BATCH_SIZE),
                           use_multiprocessing = False,
                           verbose = 1,
                           workers=1
                           )


print('Test loss: ' , loss)
print('Test accuracy: ', accuracy)
print('Test f1_score: ', f1_score)
print('Test precision: ', precision)
print('Test recall: ', recall)

#%%

#from sklearn.metrics import  confusion_matrix
#Y_pred = model.predict_generator(eval_generator, np.ceil(len(eval_generator) / BATCH_SIZE))
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#cf = confusion_matrix(eval_generator.classes, y_pred)
#print(cf)


#df_cm = pd.DataFrame(cf, index = [i for i in ['normal', 'pneumonia', 'Covid-19']],
#                  columns = [i for i in ['normal', 'pneumonia', 'Covid-19']])
#plt.figure(figsize = (10,7))
#sn.heatmap(df_cm, annot=True)
#plt.savefig('./plots/confusion_matrix.png', dpi=300)
