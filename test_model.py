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
import argparse
from pathlib import Path
from Utils.utilities import load_imgs
from testing.testing import *
from models.loss_functions import *
from models.models import *

# import U-net and loss functions
s = Semantic_loss_functions()
m = Unet_models()

# Define paths
parser = argparse.ArgumentParser()
parser.add_argument("test_img_path")
parser.add_argument("save_path")
parser.add_argument("model_path")
args = parser.parse_args()


test_path = Path(args.test_img_path)
if not test_path.exists():
    logger.error("The test img path dosen't exist")
    raise SystemExit(1)

model_path = Path(args.model_path)
if not model_path.exists():
    logger.error("The model img path dosen't exist")
    raise SystemExit(1)

save_path = Path(args.save_path)
if not save_path.exists():
    logger.error("The save path dosen't exist")
    raise SystemExit(1)

# Define parameters
img_size = 256
x_test, y_test,test_names = load_imgs(img_size, test_path,img_color=False)

# Remove .jpg at end of the image
for i in range(len(test_names)):
    aux_names = test_names[i]
    test_names[i] = aux_names[:-4]

test_model(x_test,y_test,800,test_names, model_load = True , model_path = model_path, save_img=True, save_imgpath=save_path)

sys.exit()