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

from utils.t_SNE_plot import*
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
                init_var = 1/(28.*28.)
            else:#mask
                init_var = 1/4.
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
                loss += precision * 28*7*K.sum(K.square(y_true - y_pred) ) + log_var[0] #*128*32
            else:#mask
                loss += precision * K.sum(K.square(y_true - y_pred)) + log_var[0]
                #loss += K.mean(precision * (y_true - y_pred) ** 2.) + log_var[0]
            print('precision ', precision)
            print('log_var ', log_var[0])
            print('batch size ', y_pred.shape)
            loss_indx += 1
        return loss, log_var

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

class DHAE(object):
    def __init__(self, img_y, img_x, box_dim, filters, embed_dim, path,
                 n_G=1, MTL=True, optimizer='adadelta', channel=1):
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
        self.last_shape_dim = 3
        #self.final_loss_mask = None
        #self.final_loss_box = None

    def final_loss_mask(self, y_true, y_pred):  # [y_t_m,y_t_b],[y_p_m,y_p_b]
        # use predicted box and mask from pretrained model
        # ignore random initialization - easier to converge
        l_m = 0.5 *K.mean(K.square(y_true - y_pred))
        return l_m

    def final_loss_box(self, y_true, y_pred):
        l_b = 0.5 *K.mean(K.square(y_true - y_pred))
        return l_b

    def build_model(self):
        m_encoded, mask_encoder = DHAE.mask_encoder(self)
        b_encoded, box_encoder = DHAE.box_encoder(self)
        concat_feature = concatenate([m_encoded, b_encoded])

        bottleneck = Dense(self.embed_dim, activation='relu',
                           input_shape=( self.mask_embed_dim+
                                         self.box_embed_dim,))(concat_feature)  # dense_2
        bottleneck_model = Model([self.input_img, self.input_box], bottleneck)

        em = Dense(self.last_shape_dim*self.last_shape_dim*self.filters[2],
                   activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        eb = Dense(self.embed_dim, activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        em = Reshape((self.last_shape_dim, self.last_shape_dim, self.filters[2]))(em)

        decoded_mask = DHAE.mask_decoder(self, em)
        decoded_box = DHAE.box_decoder(self, eb)
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
            losses_all = {'conv2d_4': self.final_loss_mask,
                          'dense_4': self.final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
            lossWeights = {'conv2d_4': 0.5, 'dense_4': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
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
        m_encoded, mask_encoder = DHAE.mask_encoder_test(self)
        b_encoded, box_encoder = DHAE.box_encoder(self)
        concat_feature = concatenate([m_encoded, b_encoded])

        bottleneck = Dense(self.embed_dim, activation='relu',
                           input_shape=( self.mask_embed_dim+
                                         self.box_embed_dim,))(concat_feature)  # dense_2
        # bottleneck = BatchNormalization()(bottleneck)
        # bottleneck = Dropout(0.5)(bottleneck)
        bottleneck_model = Model([self.input_img, self.input_box], bottleneck)

        em = Dense(self.last_shape_dim*self.last_shape_dim*self.filters[2],
                   activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        eb = Dense(self.embed_dim, activation='relu', input_shape=(self.embed_dim,))(bottleneck)
        # em = Dense(768, activation='relu',input_shape=(256,))(em)
        em = Reshape((self.last_shape_dim, self.last_shape_dim, self.filters[2]))(em)

        decoded_mask = DHAE.mask_decoder_test(self, em)
        decoded_box = DHAE.box_decoder(self, eb)
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
                optimizer = SGD(learning_rate=0.001, momentum=0.9)
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
            losses_all = {'conv2d_4': self.final_loss_mask,
                          'dense_6': self.final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
            lossWeights = {'conv2d_4': 0.5, 'dense_6': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
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
        #input dim: 28x28
        #
        #m1 = Conv2D(self.filters[0], (3, 3),
                    #activation='relu', padding='same')(self.input_img)

        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),
                    activation='relu', padding='same')(self.input_img)  # 28*28>>14*14
        # m1 = BatchNormalization()(m1)
        m1 = Dropout(0.25)(m1)

        #dim: 14x14
        #
        #m1 = Conv2D(self.filters[1], (3, 3),
                    #activation='relu', padding='same')(m1)

        m1 = Conv2D(self.filters[1], (3, 3), strides=(2, 2),
                    activation='relu', padding='same')(m1)  # 14*14>>7*7
        #m1 = BatchNormalization()(m1)
        m1 = Dropout(0.25)(m1)
        #dim: 7x7
        m1 = Conv2D(self.filters[2], (3, 3), strides=(2, 2),
                    activation='relu', padding='valid')(m1)  # 7*7>>3*3
        m1 = Flatten()(m1)
        m1 = Dense(self.mask_embed_dim, activation='relu')(m1)
        # e1 = BatchNormalization()(e1)
        # e1 = Dropout(0.25)(e1)
        return m1, Model(self.input_img, m1)

    def mask_encoder_test(self):
        # input dim: 28x28
        #
        #m1 = Conv2D(self.filters[0], (3, 3),
                   # activation='relu', padding='same')(self.input_img)
        m1 = Conv2D(self.filters[0], (3, 3), strides=(2, 2),
                    activation='relu', padding='same')(self.input_img)  # 28*28>>14*14
        #m1 = BatchNormalization()(m1)
        # dim: 14x14
        #
        #m1 = Conv2D(self.filters[1], (3, 3),
                    #activation='relu', padding='same')(m1)
        m1 = Conv2D(self.filters[1], (3, 3), strides=(2, 2),
                    activation='relu', padding='same')(m1)  # 14*14>>7*7
        #m1 = BatchNormalization()(m1)
        # dim: 7x7
        m1 = Conv2D(self.filters[2], (3, 3), strides=(2, 2),
                    activation='relu', padding='valid')(m1)  # 7*7>>3*3

        m1 = Flatten()(m1)
        m1 = Dense(self.mask_embed_dim, activation='relu')(m1)
        # e1 = BatchNormalization()(e1)

        return m1, Model(self.input_img, m1)

    def box_encoder(self):
        b2 = Dense(self.box_embed_dim, activation='relu', input_shape=(self.box_dim,))(self.input_box)  # dense_1
        return b2, Model(self.input_box, b2)

    def mask_decoder(self,m_encoded):

        x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[2], 3, 3),
                            activation='relu', strides=(2, 2), padding="valid")(m_encoded)  # 3x3>>7x7
        x = Dropout(0.25)(x)
        #x = BatchNormalization()(x)
        #
        #x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[1], 7, 7),
                            #activation='relu', padding="same")(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[1], 7, 7),
                            activation='relu', strides=(2, 2), padding="same")(x)  # 7x7>>14x14
        x = Dropout(0.25)(x)
        #x = BatchNormalization()(x)
        #
        #x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 14, 14),
                            #activation='relu', padding='same')(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 14, 14),
                            activation='relu', strides=(2, 2), padding='same')(x)  # 14x14>>28*28

        m_decoded = Conv2D(self.channel, (3, 3), activation='sigmoid', padding='same')(x)
        return m_decoded

    def mask_decoder_test(self, m_encoded):
        x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[2], 3, 3),
                            activation='relu', strides=(2, 2), padding="valid")(m_encoded)  # 3x3>>7x7
        #x = Dropout(0.25)(x)
        #x = BatchNormalization()(x)
        #
        #x = Deconvolution2D(self.filters[1], (3, 3), input_shape=(self.filters[1], 7, 7),
                            #activation='relu', padding="same")(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[1], 7, 7),
                            activation='relu', strides=(2, 2), padding="same")(x)  # 7x7>>14x14
        #x = BatchNormalization()(x)
        #
        #x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 14, 14),
                           # activation='relu', padding='same')(x)

        x = Deconvolution2D(self.filters[0], (3, 3), input_shape=(self.filters[0], 14, 14),
                            activation='relu', strides=(2, 2), padding='same')(x)  # 14x14>>28*28

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
        cp_callback = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='min')
        #cp_callback_stop = callbacks.EarlyStopping(monitor=['val_loss','train_loss'],min_delta=0.00001,patience=100,verbose=1,restore_best_weights=True,mode='min')
        #final_model.save_weights(checkpoint_path.format(epoch=0))
        # TODO: setup finetunning options (load already trained weights and start training with new epochs)
        #final_model.load_weights('/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weightv1/cp-0500.ckpt')
        final_model, \
        bottleneck_model, \
        mask_encoder_model, \
        box_encoder_model = DHAE.build_model(self)
        if self.MTL:
            #final_model.load_weights('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Mask_Instance_Clustering/final_model/model_exp/MTL/mnist_28/bmvc/var_weight/cp-1000.ckpt')
            final_model.fit([x_train_mask,x_train_box], # best parameter = [300,ThaHi 512, 0.1]
                            epochs=1000,
                            callbacks=[cp_callback],
                            batch_size=1024,
                            shuffle=True,
                            validation_split=0.1,
                            )
        if not self.MTL:
            final_model.fit([x_train_mask,x_train_box],[x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                            epochs=1000,
                            callbacks=[cp_callback],
                            batch_size=1024,
                            shuffle=True,
                            validation_split=0.1
                            )
        return final_model, bottleneck_model, mask_encoder_model, box_encoder_model

    def test_net(self):
        final_model, \
        bottleneck_model, \
        mask_encoder_model, \
        box_encoder_model = DHAE.build_test_model(self)
        return final_model, bottleneck_model, mask_encoder_model, box_encoder_model

def prepare_data_mask(x_t, normalized=False):
    #x_t = normalize(x_t)
    #cam9: 13000, 13790, cam11: 10000, 11209
    if not normalized:
        assert x_t.max()==255
        scale_img = 255.
    else:
        scale_img = 1.
    x_train = x_t[0:int(x_t.shape[0]*0.9),:,:,:].astype('float32') / scale_img
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:,:,:].astype('float32') / scale_img
    print('Training Data Preparation Done...')
    return x_train, x_test

def prepare_data_box(x_t, normalized=False):
    #x_t = normalize(x_t)
    import copy
    x_norm = copy.deepcopy(x_t.astype('float'))
    if not normalized:
        #CxCy
        x_norm[:, 0:2] = (x_t[:, 0:2] + x_t[:, 2:4]/2.)/ [256., 256.]
        #wh
        x_norm[:, 2:4] = x_t[:, 2:4] / [float(max(x_t[:, 2])), float(max(x_t[:, 3]))]
    x_train = x_norm[0:int(x_norm.shape[0]*0.9),:].astype('float32')
    x_test = x_norm[int(x_norm.shape[0]*0.9):x_norm.shape[0],:].astype('float32')
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

def visualize_reconstruction(x_test, encoded_imgs, decoded_imgs,
                             img_y,img_x,model_path, embed_dim):
    # visualize compressed encoded feature
    num_images = 20
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    minval, maxval = encoded_imgs.min(), encoded_imgs.max()
    plt.figure(figsize=(30, 4))
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(embed_dim//8, 8), cmap='hot', vmin=minval, vmax=maxval)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(model_path + 'mask_reconstruction.png', dpi=300)
    plt.close()

# Train autoencoder - Main Function
if __name__ == '__main__':
    codebase_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Mask_Instance_Clustering/'
    evaluation = 'train'
    vis_reconstruction = 0
    dataset = 'MNIST-MOT'
    isMTL =1
    # Save Checkpoints
    if dataset == 'MNIST-MOT':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        t = time.process_time()
        test = 1
        n_G = 1
        img_y = 28
        img_x = 28
        img_shape = [256., 256.]
        data_loc = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
        # sahpe change at 5 frame interval which undergoes width change but height remain fixed
        x_m = np.load(
            data_loc + '/tracking-by-animation/DHAE_feature/5Object_train_org/masks_5digits.npy',
            encoding='bytes')
        # x_m = x_m.reshape([x_m.shape[0],x_m.shape[1]*x_m.shape[2]])
        # x_m = normalize_standard(x_m/255., inverse=False)
        x_train_mask, x_test_mask = prepare_data_mask(x_m.reshape([x_m.shape[0], img_y, img_x, 1]),
                                                      normalized=False)
        print('x_train_mask: ', x_train_mask.shape)

        x_b = np.load(
            data_loc + '/tracking-by-animation/DHAE_feature/5Object_train_org/boxs_5digits.npy',
            encoding='bytes')
        # x_b = normalize_standard(x_b[:,2:6]/256., inverse=False)
        x_train_box, x_test_box = prepare_data_box(x_b[:, 2:6], normalized=False)
        # x_b_minmax_scale = preprocessing.MinMaxScaler().fit([x_b[:, 3], x_b[:, 4],x_b[:, 5],x_b[:, 6]])
        # box_minmax = np.transpose(x_b_minmax_scale.transform([x_b[:, 3], x_b[:, 4],x_b[:, 5],x_b[:, 6]]))
        # x_train_box, x_test_box = prepare_data_box(box_minmax)
        # x_train_box, x_test_box = prepare_data_box(x_b[:, 2:6])
        print('x_train_box: ', x_train_box.shape)
        # x_train_box = (x_train_box - np.mean(x_train_box)) / np.std(x_train_box)
        # x_train_mask = (x_train_mask - np.mean(x_train_mask)) / np.std(x_train_mask)
        # x_test_box = (x_test_box - np.mean(x_test_box)) / np.std(x_test_box)
        # x_test_mask = (x_test_mask - np.mean(x_test_mask)) / np.std(x_test_mask)
        assert len(x_train_box) == len(x_train_mask) and len(x_test_box) == len(x_test_mask)
        save_model_path = data_loc + '/Mask_Instance_Clustering/final_model/model_exp/'
        if isMTL:
            model_path = save_model_path + 'MTL/mnist_28/bmvc/var_weight/'
        else:
            model_path = save_model_path + 'arbit_weight/mnist_28/bmvc/var_weight'
        # clear old files
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        '''
        aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
        '''
        img_y=28
        img_x=28
        channel_num=1
        box_dim = 4
        filters =[16,24,48] # bmvc2021: current best: [16,32,48]
        embed_dim = 64

        if dataset=='MNIST-MOT' and not vis_reconstruction:
            final_model, bottleneck_model, \
            mask_encoder_model, box_encoder_model = DHAE(img_y,
                                                         img_x,
                                                         box_dim,
                                                         filters,
                                                         embed_dim,
                                                         model_path,
                                                         n_G=1,
                                                         MTL=isMTL,
                                                         optimizer='adadelta',
                                                         channel=channel_num).train(x_train_mask,
                                                                                    x_train_box)

            if isMTL:
                input_img, \
                input_box, \
                decoded_imgs, \
                decoded_box = final_model.predict([x_test_mask,x_test_box])
            else:
                decoded_imgs,decoded_box = final_model.predict([x_test_mask,x_test_box])
            #concat_feature = concatenated_model.predict([x_test_mask,x_test_box])
            #load weights into the bottleneck model
            names = [weight.name for layer in final_model.layers for weight in layer.weights]
            weights = final_model.get_weights()

            i = 0
            for name, weight in zip(names, weights):
                print(i, name, weight.shape)
                i += 1
            bottleneck_model.set_weights(final_model.get_weights()[:12])
            bottleneck_feature = bottleneck_model.predict([x_test_mask,x_test_box])
            import matplotlib.pylab as pylab
            pylab.plot(final_model.history.history['loss'])
            pylab.show()
            loss_curve(final_model, model_path)
            visualize_reconstruction(x_test_mask, bottleneck_feature, decoded_imgs,
                                     img_y,img_x,model_path, embed_dim)

            # Found Standard Deviations
            print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in final_model.layers[-1].log_vars])

            print('%f' % (time.process_time()-t))
    if dataset == 'Sprites-MOT':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        t = time.process_time()
        test = 1
        n_G = 1
        img_y = 28
        img_x = 28
        img_shape = [256., 256.]
        data_loc = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
        # sahpe change at 5 frame interval which undergoes width change but height remain fixed
        x_m = np.load(
            data_loc + '/tracking-by-animation/DHAE_feature/SPRITE-MOT/5Object_train_org/masks_5digits.npy',
            encoding='bytes')
        # x_m = x_m.reshape([x_m.shape[0],x_m.shape[1]*x_m.shape[2]])
        # x_m = normalize_standard(x_m/255., inverse=False)
        x_train_mask, x_test_mask = prepare_data_mask(x_m.reshape([x_m.shape[0], img_y, img_x, 1]),
                                                      normalized=False)
        print('x_train_mask: ', x_train_mask.shape)

        x_b = np.load(
            data_loc + '/tracking-by-animation/DHAE_feature/SPRITE-MOT/5Object_train_org/boxs_5digits.npy',
            encoding='bytes')
        # x_b = normalize_standard(x_b[:,2:6]/256., inverse=False)
        x_train_box, x_test_box = prepare_data_box(x_b[:, 2:6], normalized=False)
        # x_b_minmax_scale = preprocessing.MinMaxScaler().fit([x_b[:, 3], x_b[:, 4],x_b[:, 5],x_b[:, 6]])
        # box_minmax = np.transpose(x_b_minmax_scale.transform([x_b[:, 3], x_b[:, 4],x_b[:, 5],x_b[:, 6]]))
        # x_train_box, x_test_box = prepare_data_box(box_minmax)
        # x_train_box, x_test_box = prepare_data_box(x_b[:, 2:6])
        print('x_train_box: ', x_train_box.shape)
        # x_train_box = (x_train_box - np.mean(x_train_box)) / np.std(x_train_box)
        # x_train_mask = (x_train_mask - np.mean(x_train_mask)) / np.std(x_train_mask)
        # x_test_box = (x_test_box - np.mean(x_test_box)) / np.std(x_test_box)
        # x_test_mask = (x_test_mask - np.mean(x_test_mask)) / np.std(x_test_mask)
        assert len(x_train_box) == len(x_train_mask) and len(x_test_box) == len(x_test_mask)
        save_model_path = data_loc + '/Mask_Instance_Clustering/final_model/model_exp/'
        if isMTL:
            model_path = save_model_path + 'MTL/sprite_28/bmvc/var_weight/'
        else:
            model_path = save_model_path + 'arbit_weight/sprite_28/bmvc/var_weight'
        # clear old files
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        '''
        aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
        '''
        img_y=28
        img_x=28
        channel_num=1
        box_dim = 4
        filters = [16,32,32] # bmvc2021
        embed_dim = 64

        final_model, bottleneck_model, \
        mask_encoder_model, box_encoder_model = DHAE(img_y,
                                                     img_x,
                                                     box_dim,
                                                     filters,
                                                     embed_dim,
                                                     model_path,
                                                     n_G=1,
                                                     MTL=isMTL,
                                                     optimizer='adadelta',
                                                     channel=channel_num).train(x_train_mask,
                                                                                x_train_box)

        if isMTL:
            input_img, \
            input_box, \
            decoded_imgs, \
            decoded_box = final_model.predict([x_test_mask,x_test_box])
        else:
            decoded_imgs,decoded_box = final_model.predict([x_test_mask,x_test_box])
        #concat_feature = concatenated_model.predict([x_test_mask,x_test_box])
        #load weights into the bottleneck model
        names = [weight.name for layer in final_model.layers for weight in layer.weights]
        weights = final_model.get_weights()

        i = 0
        for name, weight in zip(names, weights):
            print(i, name, weight.shape)
            i += 1
        bottleneck_model.set_weights(final_model.get_weights()[:12])
        bottleneck_feature = bottleneck_model.predict([x_test_mask,x_test_box])
        import matplotlib.pylab as pylab
        pylab.plot(final_model.history.history['loss'])
        pylab.show()
        loss_curve(final_model, model_path)
        visualize_reconstruction(x_test_mask, bottleneck_feature, decoded_imgs,
                                 img_y,img_x,model_path, embed_dim)

        # Found Standard Deviations
        print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in final_model.layers[-1].log_vars])

        print('%f' % (time.process_time()-t))

