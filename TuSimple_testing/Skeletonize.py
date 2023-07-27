import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize , thin


def skeletonize(img):
  """ OpenCV function to return a skeletonized version of img, a Mat object"""

  img = img.copy()  # don't clobber original
  skel = img.copy()

  skel[:, :] = 0
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

  while True:
    eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img[:, :] = eroded[:, :]
    if cv2.countNonZero(img) == 0:
      break

  return skel


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

def load_imgs_dataset (img_size,img_path):
  """Load all the images 

  Args:
      img_size (int): size of the image
      img_path (string): path of the images

  Returns:
      x_input: images in the folder.
  """

  # Read all the images in the folder
  list_files, train_names = read_files(img_path)

  # Declare the variables for preallocation
  x_input = np.empty((len(list_files), img_size, img_size, 3), dtype=np.float32)
  y_input = np.empty((len(list_files), img_size, img_size,3), dtype=np.float32)

  test = enumerate(list_files)

  for i in tqdm(range(len(list_files))):
    fpath = list_files[i]
    img = cv2.imread(fpath)

    # Divide and resize input image and label image
    input_image = cv2.resize((img), (img_size,img_size), interpolation = cv2.INTER_AREA)

    # Copy the image to the preallocated variable
    x_input[i, ...] = input_image


  return x_input

def load_imgs_predictions (img_size,img_path):
  """Load and process output images from the model

  Args:
      img_size (int): size of the prediction images
      img_path (string): path of the prediction images

  Returns:
      imgs, strings: processed prediction images, names of the images
  """

  # Read all the images in the folder
  list_files, img_names = read_files(img_path)

  # Declare the variables for preallocation
  y_input = np.empty((len(list_files), img_size, img_size,3), dtype=np.float32)

  for i in tqdm(range(len(list_files))):
    fpath = list_files[i]
    img2 = cv2.imread(fpath)

    # Binarize the label image
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    th, img2 = cv2.threshold(img2, 10, 1, cv2.THRESH_BINARY)

    # Skeletonize
    img2 = thin(img2, max_iter=10)
    img2 = np.expand_dims(img2, axis=2)

    # Copy the image to the preallocated variable
    y_input[i, ...] = img2

  return y_input , img_names

def skeletonize_dataset(img_size, path_data, path_pred, save_path):
  """Apply skeletonization for the dataset

  Args:
      img_size (int): Size of the images
      path_data (string): Path of the label data
      path_pred (string): path of the predicted images
      save_path (string): Path where the images will be saved
  """
  # Load all the images
  print('Loading dataset images fot skeletonization')
  dataset_img = load_imgs_dataset(img_size,path_data)
  print('Done!')

  print('Loading prediction images')
  prediction_img , img_names = load_imgs_predictions (img_size,path_pred)
  print('Done!')

  print('Creating new images')
  for i in tqdm(range(len(prediction_img))):
    colored_prediction = (dataset_img[i]*prediction_img[i]*255)/255
    cv2.imwrite(save_path+img_names[i], colored_prediction)

  print('Done!')




