from keras.layers import Input, Dense, Conv2D,Dropout, MaxPooling2D, UpSampling2D,Deconvolution2D, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import adadelta, Adam, SGD
from keras.initializers import VarianceScaling
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Lambda
from keras.backend import slice
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import callbacks
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
#from t_SNE_plot import*
import cv2
import tensorflow as tf
from keras.initializers import Constant
#from scipy.misc import imsave
import os
import time
import sys
import glob
import math
from keras import objectives
from keras.engine.topology import Layer
#np.random.seed(10)
# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            if i==1:#box
                init_var = np.log(1/4.)  # reported: 1/(128.*128.)
            else:#mask
                init_var = np.log(1/(128.*128.))#reported: 1/4.
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(init_var), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = 0.5*K.exp(-log_var[0])
            if loss_indx == 1:#box
                loss += K.mean(precision * K.square(y_true - y_pred)) + log_var[0] #*128*32
            else:#masked RGB
                #loss += K.sum(precision * K.square(y_true - y_pred)) + log_var[0]
                loss += K.mean(precision * (y_true - y_pred) ** 2. , -1) + log_var[0]
            print('precision ', precision)
            print('log_var ', log_var[0])
            print('batch size ', y_pred.shape)
            loss_indx += 1
        return loss, log_var

    def multi_loss_exp(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        for y_true, y_pred, var in zip(ys_true, ys_pred, self.log_vars):
            precision = 0.5*(1/var[0]**2) #K.exp(-log_var[0])
            if loss_indx == 1:#box
                loss += K.mean(precision * K.square(y_true - y_pred)) + K.log(var[0]) #*128*32
            else:#masked RGB
                #loss += K.sum(precision * K.square(y_true - y_pred)) + log_var[0]
                loss += K.mean(precision * (y_true - y_pred) ** 2. , -1) + K.log(var[0])
            print('precision ', precision)
            print('var ', var[0])
            print('batch size ', y_pred.shape)
            loss_indx += 1
        return loss, var

    def call(self, inputs):
        # TODO: we use the inputs (x_m,x_b) as predicted outputs (y_m,y_b) and model output (f^W([x_m,x_b])) as y_m,y_b
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss, log_var = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        print('log_var ', log_var[0])
        # We won't actually use the output.
        # return K.concatenate(inputs, -1)
        # TODO: return inputs or outputs?? - output will be the inputs
        return inputs

class DHAE_pRGB(object):
    def __init__(self, img_y, img_x, box_dim, filters, embed_dim,
                 path, n_G=1, MTL=True, optimizer='adam', channel=3):
        self.input_img = Input(shape=(img_y, img_x, channel))
        self.input_box = Input(shape=(box_dim,))
        self.optimizer_type = optimizer
        self.filters = filters # filteers = [16,32,48]
        self.channel = channel
        self.mask_embed_dim = embed_dim
        self.box_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.box_dim = box_dim
        self.MTL = MTL
        self.GPUS = n_G
        self.model_path = path
        self.last_shape_dim = 4
        #self.final_loss_mask = None
        #self.final_loss_box = None

    def final_loss_mask(self, y_true, y_pred):  # [y_t_m,y_t_b],[y_p_m,y_p_b]
        # use predicted box and mask from pretrained model
        # ignore random initialization - easier to converge
        l_m = 0.5 *K.mean(K.square(y_true - y_pred))
        return l_m

    def final_loss_box(self, y_true, y_pred):
        l_b = 0.5 *K.mean(K.square(y_true - y_pred)) #128*32
        return l_b

    def build_model(self):
        m_encoded, mask_encoder = DHAE_pRGB.mask_encoder(self)
        b_encoded, box_encoder = DHAE_pRGB.box_encoder(self)
        concat_feature = concatenate([m_encoded, b_encoded])

        bottleneck = Dense(self.embed_dim, activation='relu', input_shape=( self.mask_embed_dim+ self.box_embed_dim,))(concat_feature)  # dense_2
        # bottleneck = BatchNormalization()(bottleneck)
        # bottleneck = Dropout(0.5)(bottleneck)
        bottleneck_model = Model([self.input_img, self.input_box], bottleneck)

        em = Dense(self.last_shape_dim*self.last_shape_dim*self.filters[3], activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        eb = Dense(self.embed_dim, activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        # em = Dense(768, activation='relu',input_shape=(256,))(em)
        em = Reshape((self.last_shape_dim, self.last_shape_dim, self.filters[3]))(em)

        decoded_mask = DHAE_pRGB.mask_decoder(self, em)
        decoded_box = DHAE_pRGB.box_decoder(self, eb)
        # out = CustomMultiLossLayer(nb_outputs=2)([ym_true, yb_true, ym_pred, yb_pred])
        if self.MTL:
            out = CustomMultiLossLayer(nb_outputs=2)([self.input_img, self.input_box, decoded_mask, decoded_box])
            # check to see if we are compiling using just a single GPU
            if self.GPUS <= 1:
                print("[INFO] training with 1 GPU...")
                final_model = Model([self.input_img, self.input_box], out)
            # otherwise, we are compiling using multiple GPUs
            else: #TODO: setup multi-GPU training options
                print("[INFO] training with {} GPUs...".format(self.GPUS))
                # TODO: implement multigpu capability
                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    # initialize the model
                    final_model = Model([self.input_img, self.input_box], out)

            if self.optimizer_type == 'adam':
                optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
            if self.optimizer_type == 'sgd':
                optimizer = SGD(learning_rate=0.00001, momentum=0.9)
            if self.optimizer_type == 'adadelta':
                optimizer = 'adadelta'
            final_model.compile(optimizer=optimizer, loss=None)
            print('model is builded with optimizer {}'.format(optimizer))
            # final_model.compile(optimizer='adadelta', loss=None)
        # When arbitrary weights are used for multi-tasking
        if not self.MTL:
            # check to see if we are compiling using just a single GPU
            if self.GPUS <= 1:
                final_model = Model([self.input_img, self.input_box], [decoded_mask, decoded_box])
            else:
                print("[INFO] training with {} GPUs...".format(self.GPUS))
                with tf.device("/cpu:0"):
                    final_model = Model([self.input_img, self.input_box], [decoded_mask, decoded_box])
            # set loss function for arbitrary weight
            print('Model Training With Arbitrary Multi-Task Weights.....')
            losses_all = {'conv2d_6': self.final_loss_mask,
                          'dense_6': self.final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
            lossWeights = {'conv2d_6': 0.5, 'dense_6': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
            # opt = adadelta(lr=ini_lr,decay=ini_lr/epochs)
            # (lr=0.001, decay=.001 / EPOCHS)
            if self.optimizer_type == 'sgd':
                optimizer = SGD(learning_rate=0.00001, momentum=0.9)
            if self.optimizer_type == 'adam':
                optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
            if self.optimizer_type == 'adadelta':
                optimizer = 'adadelta'
            final_model.compile(optimizer=optimizer, loss=losses_all, loss_weights=lossWeights)

        print(final_model.summary())
        # assert len(final_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
        # assert len(final_model.losses) == 1
        return final_model, bottleneck_model, mask_encoder, box_encoder

    def build_test_model(self):
        m_encoded, mask_encoder = DHAE_pRGB.mask_encoder_test(self)
        b_encoded, box_encoder = DHAE_pRGB.box_encoder(self)
        concat_feature = concatenate([m_encoded, b_encoded])

        bottleneck = Dense(self.embed_dim, activation='relu', input_shape=( self.mask_embed_dim+ self.box_embed_dim,))(concat_feature)  # dense_2
        # bottleneck = BatchNormalization()(bottleneck)
        # bottleneck = Dropout(0.5)(bottleneck)
        bottleneck_model = Model([self.input_img, self.input_box], bottleneck)

        em = Dense(self.last_shape_dim*self.last_shape_dim*self.filters[3], activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        eb = Dense(self.embed_dim, activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        # em = Dense(768, activation='relu',input_shape=(256,))(em)
        em = Reshape((self.last_shape_dim, self.last_shape_dim, self.filters[3]))(em)

        decoded_mask = DHAE_pRGB.mask_decoder_test(self, em)
        decoded_box = DHAE_pRGB.box_decoder(self, eb)
        # out = CustomMultiLossLayer(nb_outputs=2)([ym_true, yb_true, ym_pred, yb_pred])
        if self.MTL:
            out = CustomMultiLossLayer(nb_outputs=2)([self.input_img, self.input_box, decoded_mask, decoded_box])
            # check to see if we are compiling using just a single GPU
            if self.GPUS <= 1:
                print("[INFO] training with 1 GPU...")
                final_model = Model([self.input_img, self.input_box], out)
            # otherwise, we are compiling using multiple GPUs
            else: #TODO: setup multi-GPU training options
                print("[INFO] training with {} GPUs...".format(self.GPUS))
                # TODO: implement multigpu capability
                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    # initialize the model
                    final_model = Model([self.input_img, self.input_box], out)
            if self.optimizer_type == 'adam':
                optimizer =  Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
            if self.optimizer_type == 'sgd':
                optimizer = SGD(learning_rate=0.00001, momentum=0.9)
            if self.optimizer_type == 'adadelta':
                optimizer = 'adadelta'
            final_model.compile(optimizer=optimizer, loss=None)
            # final_model.compile(optimizer='adadelta', loss=None)
        # When arbitrary weights are used for multi-tasking
        if not self.MTL:
            # check to see if we are compiling using just a single GPU
            if self.GPUS <= 1:
                final_model = Model([self.input_img, self.input_box], [decoded_mask, decoded_box])
            else:
                print("[INFO] training with {} GPUs...".format(self.GPUS))
                with tf.device("/cpu:0"):
                    final_model = Model([self.input_img, self.input_box], [decoded_mask, decoded_box])
            # set loss function for arbitrary weight
            print('Model Training With Arbitrary Multi-Task Weights.....')
            losses_all = {'conv2d_6': self.final_loss_mask,
                          'dense_6': self.final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
            lossWeights = {'conv2d_6': 0.5, 'dense_6': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
            # opt = adadelta(lr=ini_lr,decay=ini_lr/epochs)
            # (lr=0.001, decay=.001 / EPOCHS)
            if self.optimizer_type == 'sgd':
                optimizer = SGD(learning_rate=0.001, momentum=0.9)
            if self.optimizer_type == 'adam':
                optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
            if self.optimizer_type == 'adadelta':
                optimizer = 'adadelta'
            final_model.compile(optimizer=optimizer, loss=losses_all, loss_weights=lossWeights)

        print(final_model.summary())
        # assert len(final_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
        # assert len(final_model.losses) == 1
        return final_model, bottleneck_model, mask_encoder, box_encoder

    def mask_encoder(self):
        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(self.input_img)  # 128*128>>64*64
        #m1 = BatchNormalization()(m1)

        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 64*64>>32*32
        m1 = BatchNormalization()(m1)
        m1 = Dropout(0.25)(m1)

        m1 = Conv2D(self.filters[1], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 32*32>>16*16
        m1 = BatchNormalization()(m1)
        m1 = Dropout(0.25)(m1)

        m1 = Conv2D(self.filters[2], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 16*16>>8*8
        m1 = BatchNormalization()(m1)
        m1 = Dropout(0.25)(m1)
        m1 = Conv2D(self.filters[3], (3, 3), strides=(2, 2), activation='relu',data_format="channels_last", padding='same')(m1)  # 8*8>>4*4
        # e1 = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)  #
        # e1 = BatchNormalization()(e1)
        # e1 = Dropout(0.25)(e1)
        #e1 = Reshape((768,))(e1)
        m1 = Flatten()(m1)
        m1 = Dense(self.mask_embed_dim, activation='relu')(m1)
        # e1 = BatchNormalization()(e1)
        # e1 = Dropout(0.25)(e1)
        return m1, Model(self.input_img, m1)

    def mask_encoder_test(self):
        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(self.input_img)  # 128*128>>64*64
        #m1 = BatchNormalization()(m1)

        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 64*64>>32*32
        m1 = BatchNormalization()(m1)

        m1 = Conv2D(self.filters[1], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 32*32>>16*16
        m1 = BatchNormalization()(m1)

        m1 = Conv2D(self.filters[2], (3, 3), strides=(2, 2),data_format="channels_last", activation='relu', padding='same')(m1)  # 16*16>>8*8
        m1 = BatchNormalization()(m1)

        m1 = Conv2D(self.filters[3], (3, 3), strides=(2, 2), activation='relu',data_format="channels_last", padding='same')(m1)  # 8*8>>4*4
        # e1 = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)  #
        # e1 = BatchNormalization()(e1)
        # e1 = Dropout(0.25)(e1)
        #e1 = Reshape((768,))(e1)
        m1 = Flatten()(m1)
        m1 = Dense(self.mask_embed_dim, activation='relu')(m1)
        # e1 = BatchNormalization()(e1)

        return m1, Model(self.input_img, m1)

    def box_encoder(self):
        b2 = Dense(self.box_embed_dim, activation='relu', input_shape=(self.box_dim,))(self.input_box)  # dense_1
        return b2, Model(self.input_box, b2)

    def mask_decoder(self,m_encoded):
        # new_rows = ((rows - 1) * strides[0] + kernel_size[0]- 2 * padding[0] + output_padding[0])
        # x = Deconvolution2D(8, (3, 3), input_shape=(8, 2, 2), activation='relu', strides=(2, 2), padding="same")(encoded)#out-4
        # x = BatchNormalization()(x)
        # x = Dropout(0.25)(x)
        x = Deconvolution2D(self.filters[3], (3, 3), input_shape=(self.filters[2], 4, 4), activation='relu', strides=(2, 2), padding="same")(m_encoded)  # 4*4>>8*8
        x = Dropout(0.25)(x)
        #x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[2], (3, 3), input_shape=(self.filters[1], 8, 8), activation='relu', strides=(2, 2), padding="same")(x)  # 8*8>>16*16
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[1], 16, 16), activation='relu', strides=(2, 2), padding='same')(x)  # 16*16>>32*32
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 32, 32), activation='relu', strides=(2, 2), padding='same')(x)  # 32*32>>64*64
        #x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 64, 64), activation='relu', strides=(2, 2), padding='same')(x)  # 64*64>>128*128
        #x = BatchNormalization()(x)
        m_decoded = Conv2D(self.channel, (3, 3), activation='sigmoid', padding='same')(x)
        return m_decoded

    def mask_decoder_test(self,m_encoded):
        # new_rows = ((rows - 1) * strides[0] + kernel_size[0]- 2 * padding[0] + output_padding[0])
        # x = Deconvolution2D(8, (3, 3), input_shape=(8, 2, 2), activation='relu', strides=(2, 2), padding="same")(encoded)#out-4
        # x = BatchNormalization()(x)
        # x = Dropout(0.25)(x)
        x = Deconvolution2D(self.filters[3], (3, 3), input_shape=(self.filters[2], 4, 4), activation='relu', strides=(2, 2), padding="same")(m_encoded)  # 4*4>>8*8
        #x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[2], (3, 3), input_shape=(self.filters[1], 8, 8), activation='relu', strides=(2, 2), padding="same")(x)  # 8*8>>16*16
        x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[1], 16, 16), activation='relu', strides=(2, 2), padding='same')(x)  # 16*16>>32*32
        x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 32, 32), activation='relu', strides=(2, 2), padding='same')(x)  # 32*32>>64*64
        #x = BatchNormalization()(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 64, 64), activation='relu', strides=(2, 2), padding='same')(x)  # 64*64>>128*128
        #x = BatchNormalization()(x)
        m_decoded = Conv2D(self.channel, (3, 3), activation='sigmoid', padding='same')(x)
        return m_decoded

    def box_decoder(self,b_encoded):
        # TODO:
        b_decoded = Dense(self.box_dim, activation='sigmoid', input_shape=(self.embed_dim,))(b_encoded)  # dense_4
        return b_decoded

    def train(self,x_train_mask,x_train_box):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        checkpoint_path = self.model_path + 'cp-{epoch:04d}.ckpt'#'final_model.cpt'#'cp-{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #cp_callback = callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_best_only=True,save_weights_only=True,period=10,)
        #es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        #TODO: Set proper early stopping criterion
        cp_callback = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        #cp_callback_stop = callbacks.EarlyStopping(monitor=['val_loss','train_loss'],min_delta=0.00001,patience=100,verbose=1,restore_best_weights=True,mode='min')
        #final_model.save_weights(checkpoint_path.format(epoch=0))
        # TODO: setup finetunning options (load already trained weights and start training with new epochs)
        #final_model.load_weights('/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weightv1/cp-0500.ckpt')
        final_model, bottleneck_model, mask_encoder_model, box_encoder_model = DHAE_pRGB.build_model(self)
        if self.MTL:
            final_model.fit([x_train_mask,x_train_box], # best parameter = [300,ThaHi 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=512,
                            shuffle=True,
                            validation_split=0.1,
                            )
        if not self.MTL:
            final_model.fit([x_train_mask,x_train_box],[x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=512,
                            shuffle=True,
                            validation_split=0.1
                            )
        return final_model, bottleneck_model, mask_encoder_model, box_encoder_model

    def test_net(self):
        final_model, \
        bottleneck_model, \
        mask_encoder_model, \
        box_encoder_model = DHAE_pRGB.build_test_model(self)
        return final_model, bottleneck_model, mask_encoder_model, box_encoder_model

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def prepare_data_mask(x_t):
    #x_t = normalize(x_t)
    x_train = x_t[0:int(x_t.shape[0]*0.9),:,:,:].astype('float32') / 255. #normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:,:,:].astype('float32') / 255.
    print('Training Data Preparation Done...')
    return x_train, x_test

def prepare_data_box(x_t):
    #x_t = normalize(x_t)
    x_train = x_t[0:int(x_t.shape[0]*0.9),:].astype('float32')# / 255. box#normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:].astype('float32')# / 255.
    print('Training Data Preparation Done...')
    return x_train, x_test

def loss_curve(autoencoder_model,model_path):
    #plt.figure(figsize=(30, 4))
    loss = autoencoder_model.history.history['loss']
    val_loss = autoencoder_model.history.history['val_loss']
    epochs = range(1,len(loss)+1)
    plt.plot(epochs, loss, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, color='green', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('model loss')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(model_path + 'loss_combined_final.png', dpi=300)
    plt.close()

def mask_loss_curve(autoencoder_model):
    plt.figure()
    loss = autoencoder_model.history.history['conv2d_transpose_7_loss']
    val_loss = autoencoder_model.history.history['val_conv2d_transpose_7_loss']
    epochs = range(1,len(loss)+1)
    plt.plot(epochs, loss, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, color='green', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('mask loss')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig('/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weight_mask_reshape/result/loss_mask.png', dpi=300)
    plt.close()

def box_loss_curve(autoencoder_model):
    plt.figure()
    loss = autoencoder_model.history.history['dense_5_loss']
    val_loss = autoencoder_model.history.history['val_dense_5_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, color='green', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('box loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(
        '/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/final_model/checkpoints_weight_mask_reshape128/loss_box.png',
        dpi=300)
    plt.close()

def visualize_reconstruction(x_test, encoded_imgs, decoded_imgs,img_y,img_x,model_path):
    # visualize compressed encoded feature
    num_images = 20
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    minval, maxval = encoded_imgs.min(), encoded_imgs.max()
    plt.figure(figsize=(30, 4))
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[image_idx].reshape(img_y, img_x,x_test.shape[3]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(16, 8), cmap='hot', vmin=minval, vmax=maxval)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(img_y, img_x,decoded_imgs.shape[3]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(model_path + 'mask_reconstruction.png', dpi=300)
    plt.close()

# Train autoencoder - Main Function
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    codebase_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Mask_Instance_Clustering'
    evaluation = 'train'
    vis_reconstruction = 0
    dataset = 'kitti_rgb'
    isMTL = 1
    # Save Checkpoints
    if evaluation == 'train':
        t = time.process_time()
        if dataset=='kitti_shape':
            x_m = np.load(codebase_path + '/train_data_all/train_mask_mot_all0_reshape128_kitti.npy',encoding='bytes')
            x_b = np.load(codebase_path + '/train_data_all/train_box_mot_all0_reshape128_kitti.npy',encoding='bytes')
            x_train_mask, x_test_mask = prepare_data_mask(x_m)
            x_train_box, x_test_box = prepare_data_box(x_b)
            assert x_train_box.shape[0]==x_train_mask.shape[0],'multi-task instances should be equal but found (mask,box) > ({},{})'.format(x_train_mask.shape[0],x_train_box.shape[0])
            model_path = codebase_path + '/final_model/checkpoints_MOTS_128_kitti/'
        if dataset=='kitti_rgb':
            x_m = np.load(codebase_path+'/train_data_all/rgb/train_mask_mot_all0_reshape128_kitti_mot17.npy',encoding='bytes')
            x_b = np.load(codebase_path+'/train_data_all/rgb/train_box_mot_all0_reshape128_kitti_mot17.npy',encoding='bytes')
            x_train_mask, x_test_mask = prepare_data_mask(x_m)
            x_train_box, x_test_box = prepare_data_box(x_b)
            assert x_train_box.shape[0]==x_train_mask.shape[0],'multi-task instances should be equal but found (mask,box) > ({},{})'.format(x_train_mask.shape[0],x_train_box.shape[0])
            #model_path = codebase_path+'/final_model/checkpoints_MOTS_128_kitti_appearance128/'
            #new exp: bmvc
            if isMTL:
                model_path = codebase_path + '/final_model/model_exp/MTL/kitti_128/bmvc/var_weight_henry/'
            else:
                model_path = codebase_path + '/final_model/model_exp/arbit_weight/kitti_128/var_wight/'



        '''
        aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
        '''
        img_y=128
        img_x=128
        channel_num = 3
        box_dim = 4
        filters = [16,32,32,64] # iccv2021
        #filters = [32, 64, 64, 128]
        embed_dim = 128

        if evaluation=='train' and not vis_reconstruction:
            final_model, bottleneck_model, \
            mask_encoder_model, box_encoder_model = DHAE_pRGB(img_y,
                                                             img_x,
                                                             box_dim,
                                                             filters,
                                                             embed_dim,
                                                             model_path,
                                                             n_G=1,
                                                             MTL=isMTL,
                                                             optimizer='adadelta',
                                                             channel=channel_num).train(x_train_mask,x_train_box)
            if not isMTL:
                decoded_imgs,decoded_box = final_model.predict([x_test_mask,x_test_box])
            else:
                input_img, \
                input_box, \
                decoded_imgs, \
                decoded_box = final_model.predict([x_test_mask,x_test_box])
            #concat_feature = concatenated_model.predict([x_test_mask,x_test_box])
            #load weights into the bottleneck model
            bottleneck_model.set_weights(final_model.get_weights()[:28])
            bottleneck_feature = bottleneck_model.predict([x_test_mask,x_test_box])
            import matplotlib.pylab as pylab
            pylab.plot(final_model.history.history['loss'])
            pylab.show()
            loss_curve(final_model,model_path)
            visualize_reconstruction(x_test_mask, bottleneck_feature, decoded_imgs,img_y,img_x,model_path)

            # Found Standard Deviations
            print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in final_model.layers[-1].log_vars])

            print('%f' % (time.process_time()-t))
    if vis_reconstruction:
        #load test_net
        checkpoint_path = model_path + 'cp-1000.ckpt'
        final_model, bottleneck_model, mask_encoder_model, \
        box_encoder_model = DHAE_pRGB(img_y,
                                      img_x,
                                      box_dim,
                                      filters,
                                      embed_dim,
                                      checkpoint_path,
                                      n_G=1,
                                      MTL=True).test_net()
        print('load checkpoint from {}'.format(checkpoint_path))
        final_model.load_weights(checkpoint_path)
        input_img, input_box, decoded_imgs, decoded_box = final_model.predict([x_test_mask, x_test_box])
        bottleneck_feature = bottleneck_model.predict([x_test_mask, x_test_box])
        visualize_reconstruction(x_test_mask,
                                 bottleneck_feature,
                                 decoded_imgs,
                                 img_y,
                                 img_x,
                                 model_path)
        print('reconstruction of random test images ae saved at {}'.format(model_path))



