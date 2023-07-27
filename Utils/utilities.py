import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

def read_files(path):
  '''
  Function for reading all the folders in the directory

  path: path of the directory
  list_files: a list of all the files in that directory.
  '''

  files = os.listdir(path)
  list_files = []
  name_files = []

  # for each folder in the path, joins the directory and saves it in a list
  for file in files:
    name_files.append(file)
    new_file = os.path.join(path, file)
    list_files.append(new_file)

  return list_files, name_files

def load_imgs (img_size,img_path,img_color = False):
  """Loads and pre-process images to the model

  Args:
      img_size (int): size of the image to process
      img_path (str): path where the images are
      img_color (bool, optional): True for color images, false for b/w images. Defaults to False.

  Returns:
      _type_: _description_
  """
  if not img_color:
    img_channel = 1
  else:
    img_channel = 3

  # Read all the images in the folder
  list_files, names = read_files(img_path)

  # Declare the variables for preallocation
  x_input = np.empty((len(list_files), img_size, img_size,img_channel), dtype=np.float32)
  y_input = np.empty((len(list_files), img_size, img_size,1), dtype=np.float32)

  for i in tqdm(range(len(list_files))):
    fpath = list_files[i]
    img = cv2.imread(fpath)

    # Obtain the half of the image
    w = img.shape[1]
    w = w // 2

    # Divide and resize input image and label image
    input_image = cv2.resize((img[:, w:, :]), (img_size,img_size), interpolation = cv2.INTER_AREA)
    label_image = cv2.resize((img[:, :w, :]), (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Convert to BW the input image
    if not img_color:
      input_image  = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
      input_image= np.expand_dims(input_image, axis=2)

    # Binarize the label image
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
    th, label_image = cv2.threshold(label_image, 10, 1, cv2.THRESH_BINARY)
    label_image = np.expand_dims(label_image, axis=2)

    # Copy the image to the preallocated variable
    x_input[i, ...] = input_image
    y_input[i, ...] = label_image


  return x_input, y_input , names


