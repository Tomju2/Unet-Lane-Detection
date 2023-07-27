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
from tqdm import tqdm
import sys

from Utils.utilities import load_imgs
from testing.testing import *

# import U-net and loss functions
from models.loss_functions import *
from models.models import *

s = Semantic_loss_functions()
m = Unet_models()

# Define parameters
img_size = 256

train_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\Training'
test_path = 'D:\Maestria\Investigacion\Datasets\TuSimple\\512x512c\\test'
x_test, y_test,test_names = load_imgs(img_size, test_path,img_color=False)

model_path = 'D:\Maestria\Checkpoint\\modelo_unet_log_cosh_dice_loss_l1_1ch_256_monog_test.h5'
save_path = 'D:\Maestria\Checkpoint\Prediccion_512'


# Remove .jpg at end of the image
for i in range(len(test_names)):
    aux_names = test_names[i]
    test_names[i] = aux_names[:-4]

test_model(x_test,y_test,800,test_names, model_load = True , model_path = model_path, save_img=True, save_imgpath=save_path)

# TuSimple testing
from TuSimple_Testing import TuSimple_test

path_data = r'D:\Maestria\Investigacion\Datasets\TuSimple\Dark_Channel\Test\\'
path_pred = r'D:\Maestria\Checkpoint\Prediccion_512\\'
save_path = r'D:\Maestria\Checkpoint\prediccion_color_skeleton\\'

pred_json_path =r'C:\Users\Tomju\PycharmProjects\Tusimple_evaluation\data\sample.json'
test_json_path =r'C:\Users\Tomju\PycharmProjects\Tusimple_evaluation\data\test_label.json'


TuSimple_test.tusimple_eval(img_size, path_data, path_pred, save_path, pred_json_path,test_json_path)

sys.exit()