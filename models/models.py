from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from glob import glob
from keras import regularizers


class Unet_models(object):
    def __init__(self):
        print("Models initialized")

    # Small u-net
    def get_small_unet_no_pool(self):
        input_layer = Input(shape=[256, 256, 1])
        c1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        l = Conv2D(filters=8, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(c1)
        c2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(l)
        l = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(c2)
        c3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(l)
        l = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(c3)
        c4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(c4), c3], axis=-1)
        l = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(l), c2], axis=-1)
        l = Conv2D(filters=24, kernel_size=(2, 2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2, 2))(l), c1], axis=-1)
        l = Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same')(l)
        l = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(l)
        l = Dropout(0.5)(l)
        output_layer = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(l)
        model = Model(input_layer, output_layer)
        return model

    # Medium u-net
    def get_med_unet(self,pretrained_weights=None):
        print('Begining Unet Small')
        weight = 32
        nb_filter = [weight, weight * 2, weight * 4, weight * 8, weight * 16]
        inputs = Input(shape=(256,256,1))
        conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5),conv4], axis=3)
        conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6),conv3], axis=3)
        conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),conv2], axis=3)
        conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8),conv1], axis=3)
        conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(3, (1, 1), activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    # W-net
    def get_unetw(self,x_train,pretrained_weights=None):
        print('Begining UNet Wide')

        weight=38
        nb_filter = [weight,weight*2,weight*4,weight*8,weight*16]
        inputs = Input(shape=x_train.shape[1:])
        conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    # CIO u-net
    def get_cio_unet(self,size = 256,input_channel = 3):

      inputs = Input((size , size , input_channel))
      conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
      conv1 = BatchNormalization() (conv1)
      conv1 = Dropout(0.1) (conv1)
      conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
      conv1 = BatchNormalization() (conv1)
      pooling1 = MaxPooling2D((2, 2)) (conv1)
      conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling1)
      conv2 = BatchNormalization() (conv2)
      conv2 = Dropout(0.1) (conv2)
      conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
      conv2 = BatchNormalization() (conv2)
      pooling2 = MaxPooling2D((2, 2)) (conv2)
      conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling2)
      conv3 = BatchNormalization() (conv3)
      conv3 = Dropout(0.2) (conv3)
      conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
      conv3 = BatchNormalization() (conv3)
      pooling3 = MaxPooling2D((2, 2)) (conv3)
      conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling3)
      conv4 = BatchNormalization() (conv4)
      conv4 = Dropout(0.2) (conv4)
      conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
      conv4 = BatchNormalization() (conv4)
      pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)
      conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling4)
      conv5 = BatchNormalization() (conv5)
      conv5 = Dropout(0.3) (conv5)
      conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
      conv5 = BatchNormalization() (conv5)
      upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
      upsample6 = concatenate([upsample6, conv4])
      conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample6)
      conv6 = BatchNormalization() (conv6)
      conv6 = Dropout(0.2) (conv6)
      conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv6)
      conv6 = BatchNormalization() (conv6)
      upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
      upsample7 = concatenate([upsample7, conv3])
      conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample7)
      conv7 = BatchNormalization() (conv7)
      conv7 = Dropout(0.2) (conv7)
      conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv7)
      conv7 = BatchNormalization() (conv7)
      upsample8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
      upsample8 = concatenate([upsample8, conv2])
      conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample8)
      conv8 = BatchNormalization() (conv8)
      conv8 = Dropout(0.1) (conv8)
      conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv8)
      conv8 = BatchNormalization() (conv8)
      upsample9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
      upsample9 = concatenate([upsample9, conv1], axis=3)
      conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample9)
      conv9 = BatchNormalization() (conv9)
      conv9 = Dropout(0.1) (conv9)
      conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv9)
      conv9 = BatchNormalization() (conv9)
      outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

      return Model(inputs=[inputs], outputs=[outputs])

    # U-net pp
    def get_unetpp(self,num_class=2, pretrained_weights=None,  deep_supervision=True):
        print('Begining UNet ++')
        nb_filter = [32, 64, 128, 256, 512]
        img_rows = 256
        img_cols = 256
        color_type = 1
        bn_axis = 1

        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
        conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            img_input)
        conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

        conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            pool1)
        conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

        up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12',
                                padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_2)
        conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_2)

        conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            pool2)
        conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            conv3_1)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

        up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22',
                                padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_2)
        conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_2)

        up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13',
                                padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13',
                              axis=bn_axis)
        conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_3)
        conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_3)

        conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
            pool3)
        conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
            conv4_1)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

        up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32',
                                padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            conv3_2)
        conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            conv3_2)

        up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23',
                                padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23',
                              axis=bn_axis)
        conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_3)
        conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_3)

        up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14',
                                padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14',
                              axis=bn_axis)
        conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_4)
        conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_4)
        conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
            pool4)
        conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
            conv5_1)

        up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42',
                                padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
            conv4_2)
        conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
            conv4_2)

        up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33',
                                padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33',
                              axis=bn_axis)
        conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            conv3_3)
        conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
            conv3_3)

        up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24',
                                padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24',
                              axis=bn_axis)
        conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_4)
        conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
            conv2_4)

        up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15',
                                padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],
                              name='merge15', axis=bn_axis)
        conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_5)
        conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
            conv1_5)

        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid',
                                  name='output_1', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(
            conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid',
                                  name='output_2', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(
            conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid',
                                  name='output_3', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(
            conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid',
                                  name='output_4', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(
            conv1_5)

        model = Model(inputs=img_input, outputs=[nestnet_output_4])

        if (pretrained_weights):
            print ("loaded weights")
            model.load_weights(pretrained_weights)
        return model