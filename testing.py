import numpy as np
from loss_functions import *
from matplotlib import pyplot as plt
import os

s = Semantic_loss_functions()


def test_eval(y_pred, y_test, x_test, eval_size, show_img=False):
    total_size = len(y_pred)

    rand_num = np.random.randint(0, high=total_size, size=eval_size)
    dice_idx = 0
    recall = 0
    precision = 0
    f1 = 0

    for i in rand_num:

        # ---------- Binarizacion de la prediccion -------------
        y_pred1 = y_pred[i]
        y_pred1 = y_pred1[:, :, 0].astype('uint8')

        # Binarizacion Otsu

        th, y_pred1 = cv2.threshold(y_pred1, 2, 1, cv2.THRESH_OTSU)

        y_pred1 = y_pred1.astype('float32')
        y_pred1 = np.expand_dims(y_pred1, axis=2)
        # ---------- Binarizacion del label -------------

        y_true = y_test[i]
        y_true = y_true[:, :, 0].astype('uint8')
        y_true = np.expand_dims(y_true, axis=2)
        # Binarizacion otsu
        th, y_true = cv2.threshold(y_true, 2, 1, cv2.THRESH_OTSU)
        y_true = y_true.astype('float32')

        if show_img:

            disp_img = x_test[i]
            # disp_img = disp_img[:,:,0]

            display_list = [disp_img, y_true, y_pred1]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.show()

            test_pred = (disp_img + y_pred1*255)

            plt.imshow(test_pred)
            plt.show()

        recall_aux, precision_aux, f1_aux = ripple_metrics(y_true, y_pred1)

        recall += recall_aux
        precision += precision_aux
        f1 += f1_aux

        dice_idx += s.dice_coef(y_true, y_pred1)

    dice_idx = dice_idx / eval_size
    recall = recall / eval_size
    precision = precision / eval_size
    f1 = f1 / eval_size

    print(dice_idx, recall, precision, f1)

def ripple_metrics(ground_t, img_prediction):
  label = ground_t.flatten()
  label_pred = img_prediction.flatten()

  TP = 0
  TN = 0
  FP = 0
  FN = 0

  # Creacion de la matriz de confusion
  for i in range(0, len(label)):
    if (label[i] == 1 and label_pred[i] == 1):
      TP = TP + 1
    elif (label[i] == 0 and label_pred[i] == 0):
      TN = TN +1
    elif (label[i] == 1 and label_pred[i] == 0):
      FP = FP + 1
    elif (label[i] == 0 and label_pred[i]) == 1:
      FN = FN + 1

  recall = TP/(FN + TP)

  precision = TP/(FP + TP)
  if (recall or precision) != 0:
    f1 = 2* ((precision * recall)/(precision + recall))
  else:
    f1 = 0

  return recall, precision, f1

def test_model(x_test,y_test,eval_size,names,model = None, model_load = False , model_path = None, save_img = False,save_imgpath = None):

    if not model_load:
        predictions = model.predict(x_test)

        test_eval(predictions, y_test, x_test, 10, show_img=True)

        test_eval(predictions, y_test, x_test, eval_size, show_img=False)
    else:
        model = tf.keras.models.load_model( model_path, custom_objects={"focal_tversky_L1" : s.focal_tversky_L1, "sensitivity":s.sensitivity, "dice_coef": s.dice_coef})
        predictions = model.predict(x_test)

        test_eval(predictions, y_test, x_test, 50, show_img=True)
        test_eval(predictions, y_test, x_test, eval_size, show_img=False)

    if save_img:
        os.chdir(save_imgpath)

        for i in range(len(predictions)):
            cv2.imwrite(names[i]+'.png', predictions[i]*255)


