import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize , thin


def skeletonize(img):
  """ OpenCV function to return a skeletonized version of img, a Mat object"""

  #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

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

  # Read all the images in the folder
  list_files, train_names = read_files(img_path)

  # Declare the variables for preallocation
  x_input = np.empty((len(list_files), img_size, img_size, 3), dtype=np.float32)
  y_input = np.empty((len(list_files), img_size, img_size,3), dtype=np.float32)

  test = enumerate(list_files)

  for i in tqdm(range(len(list_files))):
    fpath = list_files[i]
    img = cv2.imread(fpath)

    # Obtain the half of the image
    #w = img.shape[1]
    #w = w // 2

    # Divide and resize input image and label image
    input_image = cv2.resize((img), (img_size,img_size), interpolation = cv2.INTER_AREA)
    #label_image = cv2.resize((img[:, :w, :]), (img_size, img_size), interpolation=cv2.INTER_AREA)
    '''
    # Convert to BW the input image
    input_image  = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image= np.expand_dims(input_image, axis=2)

    # Binarize the label image
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
    th, label_image = cv2.threshold(label_image, 10, 1, cv2.THRESH_BINARY)
    label_image = np.expand_dims(label_image, axis=2)
    '''

    # Copy the image to the preallocated variable
    x_input[i, ...] = input_image
    #y_input[i, ...] = label_image

  return x_input

def load_imgs_predictions (img_size,img_path):

  # Read all the images in the folder
  list_files, img_names = read_files(img_path)

  # Declare the variables for preallocation
  x_input = np.empty((len(list_files), img_size, img_size,3), dtype=np.float32)

  for i in tqdm(range(len(list_files))):
    fpath = list_files[i]
    img2 = cv2.imread(fpath)

    # Binarize the label image
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    th, img2 = cv2.threshold(img2, 10, 1, cv2.THRESH_BINARY)



    # erode
    kernel = np.ones((4, 4), np.uint8)
    #img2 = cv2.erode(img2, kernel, iterations=1)

    # Skeletonize
    img2 = thin(img2, max_iter=10)
    img2 = np.expand_dims(img2, axis=2)

    # Copy the image to the preallocated variable
    x_input[i, ...] = img2

  return x_input , img_names

def skeletonize_dataset(img_size, path_data, path_pred, save_path):
  # Load all the images
  print('Loading dataset images fot skeletonization')
  dataset_img = load_imgs_dataset(img_size,path_data)
  print('Done!')

  print('Loading prediction images')
  prediction_img , img_names = load_imgs_predictions (img_size,path_pred)
  print('Done!')
  #colored_prediction = np.empty((len(prediction_img), img_size, img_size,3), dtype=np.float32)

  print('Creating new images')
  for i in tqdm(range(len(prediction_img))):
    colored_prediction = (dataset_img[i]*prediction_img[i]*255)/255
    #input_prediction = (dataset_img[i]+prediction_img[i]*255)

    cv2.imwrite(save_path+img_names[i], colored_prediction)
    #cv2.imwrite(save_path_test+ img_names[i], input_prediction)
  print('Done!')




