from keras.callbacks import *
from Utils.utilities import load_imgs
from tqdm import tqdm
import sys

# import U-net and loss functions
from models.loss_functions import *
from models.models import *
import visualkeras

s = Semantic_loss_functions()
m = Unet_models()

# -----------Load the dataset-----------------

# Define the size of the images
img_size = 256

# Define the paths
train_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\Training'
test_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\test'

# Load the images
x_train, y_train , train_names = load_imgs(img_size, train_path,img_color=False)

x_test, y_test,test_names = load_imgs(img_size, test_path,img_color=False)

x_test = x_test[0:900]
y_test = y_test[0:900]

# ---------- Create the model ---------------------

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
model =  m.get_cio_unet(input_channel= 1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-3), loss=s.focal_tversky_L1,
              metrics=[s.sensitivity, s.dice_coef])

# Establish callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_sensitivity',
                                            patience=10,
                                            verbose=1,
                                            factor=0.6,
                                            min_lr=0.000000001)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    verbose = 2,
    restore_best_weights=True)


epochs = 150
batch_size = 32
hist = model.fit(my_generator(x_train, y_train,batch_size=batch_size),
                           steps_per_epoch=40,
                           validation_data=(x_test, y_test),
                           epochs=epochs, verbose=1,
                           callbacks=[learning_rate_reduction,es])

model.save('D:\Maestria\Checkpoint\modelo_unet_log_cosh_dice_loss_l1_1ch_256_monog_test.h5')

sys.exit()
