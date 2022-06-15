import numpy as np
import pandas as pd 
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras_preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

print(os.listdir(r"Insert here"))
train_dir = r"Insert here"
test_dir = r"Insert here"
df_train = pd.read_csv(r'Insert here')
df_train.head()

# df = df_train.sample(n=10000, random_state=2020)
df = df_train # using full dataset
train, valid = train_test_split(df,test_size=0.2)

train_datagen = image.ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = image.ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x)


# We used the "flow_from_dataframe" method to create train and valid generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train,
    directory=r"Insert here",
    x_col='id',
    y_col='label',
    has_ext=False,
    batch_size=32,
    seed=2020,
    shuffle=True,
    class_mode='binary',
    target_size=(96,96))

valid_generator = test_datagen.flow_from_dataframe(
    dataframe = valid,
    directory=r"Insert here",
    x_col='id',
    y_col='label',
    has_ext=False,
    batch_size=32,
    seed=2020,
    shuffle=False,
    class_mode='binary',
    target_size=(96,96)
)

IMG_SIZE = (96, 96)
IN_SHAPE = (96, 96, 3)

dropout_dense=0.5

#must include final "input_tensor" argument
conv_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=layers.Input(shape=IN_SHAPE)
)
       
model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(dropout_dense))
model.add(layers.Dense(1, activation = "sigmoid"))

## This is the block to "freeze" certain layers, but the final result was better with no layer freezing 
# conv_base.Trainable=True
# set_trainable=False
# for layer in conv_base.layers:
#     if layer.name == 'res5a_branch2a':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

model.compile(optimizers.Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, 
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20, 
                   callbacks=[earlystopper, reducel])



############### Block to output Accuracy and Loss plots ################################
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#######################################################################################

model.save('Model.h5')