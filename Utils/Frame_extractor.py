import cv2
import os

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

def frame_extractor(path,name_file, save_path, n_frames):
    '''
    :param:
    path - path of the video
    save_path - path were the frames will be stored
    n_frames- it saves a the frame every n - frames ex: 10
    '''
    name_file = name_file[:-4]
    # create directory
    dir_path = save_path+name_file
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # open video
    video = cv2.VideoCapture(path)
    i = 0

    # While the Video is opened it runs the code
    while video.isOpened():

        ret, frame = video.read()

        # If there ir no more video it exits the loop
        if ret == False:
            break

        # it saves every nth frame
        if i % n_frames == 0:
            cv2.imwrite(dir_path + '\\'+name_file + '_'+str(i) + '.jpg', frame)
        i += 1

    print('Finished!')
    video.release()
    cv2.destroyAllWindows()


video_path = 'D:\Descargas\Videos en carretera'
save_path = 'D:\Maestria\Investigacion\Datasets\Propio\Frames\\'
n_frames = 30

list_files, name_files = read_files(video_path)

for i in range(len(list_files)):

    frame_extractor(list_files[i],name_files[i], save_path, n_frames)
