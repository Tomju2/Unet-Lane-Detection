import tensorflow as tf
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import *
from glob import glob
from keras.callbacks import *
from IPython import display
from PIL import Image


from utilities import load_imgs
from testing import *

# import U-net and loss functions
from loss_functions import *
from models import *

s = Semantic_loss_functions()
m = Unet_models()

#tf.config.list_physical_devices('GPU')

# -----------Load the dataset-----------------

# Define the size of the images
img_size = 256

# Define the paths
train_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\Training'
test_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\test'

# Load the images
x_train, y_train , train_names = load_imgs(img_size, train_path)
x_test, y_test,test_names = load_imgs(img_size, test_path)

x_test = x_test[0:800]
y_test = y_test[0:800]

# ---------- Create the model ---------------------
'''''''''
# Define image generator
def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(shear_range=0.001, fill_mode='reflect').flow(x_train, x_train, batch_size,
                                                                                     seed=50)
    mask_generator = ImageDataGenerator(shear_range=0.001, fill_mode='reflect').flow(y_train, y_train, batch_size,
                                                                                     seed=50)

    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch


# Load and compile model
model =  m.get_cio_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-3), loss=s.focal_tversky_L1,
              metrics=[s.sensitivity, s.dice_coef])

# Establish callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_dice_coef',
                                            patience=5,
                                            verbose=1,
                                            factor=0.6,
                                            min_lr=0.000000001)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    verbose = 1,
    restore_best_weights=True)

# Variables
epochs = 150
batch_size = 18


# Train model
hist = model.fit(my_generator(x_train, y_train,batch_size=batch_size),
                           steps_per_epoch=24,
                           validation_data=(x_test, y_test),
                           epochs=epochs, verbose=1,
                           callbacks=[learning_rate_reduction,es])

model.save('D:\Maestria\Checkpoint\modelo_unet_focaltversky_l1_3ch_256.h5')

'''''

model_path = 'D:\Maestria\Checkpoint\\modelo_unet_focaltversky_1ch_256.h5'
save_path = 'D:\Maestria\Checkpoint\Prediccion_512'

for i in range(len(test_names)):
    aux_names = test_names[i]
    test_names[i] = aux_names[:-4]

test_model(x_test,y_test,800,test_names, model_load = True , model_path = model_path, save_img=True, save_imgpath=save_path)

