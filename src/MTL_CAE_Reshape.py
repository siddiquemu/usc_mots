from keras.layers import Input, Dense, Conv2D,Dropout, MaxPooling2D, UpSampling2D,\
    Deconvolution2D, Flatten, Reshape, BatchNormalization, Softmax
from keras.models import Model
from keras.layers.merge import concatenate
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import adadelta,Adam
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

    @staticmethod
    def correlation_coefficient_loss(y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den

        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return 1 - K.square(r)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]

        #self.covar = [self.add_weight(name='log_corr', shape=(1,),
                                          #initializer=Constant(0.), trainable=True)]

        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss_correlation(self, ys_true, ys_pred, EmEb):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = 0.5 * K.exp(-log_var[0])
            if loss_indx == 1:  # box: L2: least square deviation 128*32*
                loss += K.sum(precision * 128 * 32 * K.square(y_true - y_pred)) + log_var[0]
            else:  # mask: L1: least absolute deviation
                loss += K.sum(precision * K.square(y_true - y_pred)) + log_var[0]
                # loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            loss_indx += 1
        # inter-task correlation
        Em = EmEb[0]
        Eb = EmEb[1]
        Tmb = self.correlation_coefficient_loss(Em, Eb)
        loss += -(K.sum(Em) + K.sum(Eb) + K.log(Tmb) \
                    - K.log(K.sum(K.exp(Em))) - K.log(K.sum(K.exp(Eb))))
        return loss, log_var

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        #http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = 0.5*K.exp(-log_var[0])
            if loss_indx==1:#box: L2: least square deviation 128*32*
                loss += K.sum(precision *128*32*K.square(y_true - y_pred)) + log_var[0]
            else:#mask: L1: least absolute deviation
                loss += K.sum(precision * K.square(y_true - y_pred)) + log_var[0]
                #loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
                #inter-task correlation

            loss_indx +=1
        return loss, log_var

    def multi_loss_dim(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        d_b = 4
        d_m = 128*128
        #http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision =K.exp(-log_var[0])#1.0/log_var[0]# K.exp(-log_var[0])
            if loss_indx==1:#box: L2: least square deviation 128*32*
                loss += K.sum(0.5*precision*128*32 * K.square(y_true - y_pred)) + log_var[0]*d_b
            else:#mask: L1: least absolute deviation
               loss += K.sum(0.5*precision * K.square(y_true - y_pred)) + log_var[0]*d_m
                #loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            loss_indx +=1
        return loss, log_var

    def multi_loss_dimv1(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        loss_indx = 0
        d_b = 4
        d_m = 128*128*2
        #http://www.chioka.in/differences-between-the-l1-norm-and-the-l2-norm-least-absolute-deviations-and-least-squares/
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision =1.0/log_var[0]# K.exp(-log_var[0])
            if loss_indx==1:#box: L2: least square deviation 128*32*
                loss += K.sum(0.5*precision * K.square(y_true - y_pred)) + K.log(log_var[0])*d_b
            else:#mask: L1: least absolute deviation
               loss += K.sum(precision * K.abs(y_true - y_pred)) + K.log(log_var[0])*d_m
                #loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            loss_indx +=1
        return loss, log_var


    def multi_loss_cov(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        recon_loss = 0
        loss_indx = 0
        log_sigma = {}
        d_m=128*128
        d_b=4
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = 0.5*K.exp(-log_var[0])
            if loss_indx==1:#box: L2: least square deviation  128*32*
                recon_loss += K.mean(precision * K.square(y_true - y_pred)) #+ log_var[0]
                cov_term_b = K.mean((y_true - y_pred))
                log_sigma['sigma_b'] =K.exp(log_var[0])**0.5
            else:#mask: L1: least absolute deviation
                recon_loss += K.mean(precision * K.square(y_true - y_pred)) #+ log_var[0]
                cov_term_m = K.mean(y_true - y_pred)
                log_sigma['sigma_m'] = K.exp(log_var[0])**0.5
            loss_indx +=1

        sigma_corr = K.exp(self.covar[0])#(self.covar[0]/(log_sigma['sigma_m']*log_sigma['sigma_b']))**2
        loss = recon_loss + \
               (1./sigma_corr)*cov_term_m*cov_term_b + \
               K.log(0.5*((log_sigma['sigma_m'])**d_m) * (log_sigma['sigma_b']-d_b*sigma_corr*log_sigma['sigma_b'])**d_b+\
               (0.5*(log_sigma['sigma_b']))**d_b * (log_sigma['sigma_m']-d_m*sigma_corr*log_sigma['sigma_m'])**d_m)
        return loss, log_var

    def call(self, inputs):
        #TODO: we use the inputs (x_m,x_b) as predicted outputs (y_m,y_b) and model output (f^W([x_m,x_b])) as y_m,y_b
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:self.nb_outputs+2]
        #EmEb = inputs[self.nb_outputs+2:]
        #loss, log_var = self.multi_loss_dim(ys_true, ys_pred)
        #loss, log_var = self.multi_loss_correlation(ys_true, ys_pred,EmEb)
        loss, log_var = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        print('log_var ', log_var[0])
        # We won't actually use the output.
        #return K.concatenate(inputs, -1)
        #TODO: return inputs or outputs?? - output will be the inputs
        return inputs[:4]

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    # class attributes
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
    # create methods
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

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
    #cam9: 13000, 13790, cam11: 10000, 11209
    x_train = x_t[0:int(x_t.shape[0]*0.9),:,:,:].astype('float32')# / 255. #normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:,:,:].astype('float32')# / 255.
    print('Training Data Preparation Done...')
    return x_train, x_test

def prepare_data_box(x_t):
    #x_t = normalize(x_t)
    #cam9: 13000, 13790, cam11: 10000, 11209
    x_train = x_t[0:int(x_t.shape[0]*0.9),:].astype('float32')# / 255. box#normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:].astype('float32')# / 255.
    print('Training Data Preparation Done...')
    return x_train, x_test
#TODO:
def final_loss_mask(y_true,y_pred):#[y_t_m,y_t_b],[y_p_m,y_p_b]
    # use predicted box and mask from pretrained model
    # ignore random initialization - easier to converge
    l_m = 0.5*K.mean(K.square(y_true - y_pred))
    return l_m

def final_loss_box(y_true,y_pred):
    l_b = 0.5*K.mean(K.square(y_true - y_pred))
    return l_b

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
        plt.imshow(x_test[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(8, 8), cmap='hot', vmin=minval, vmax=maxval)
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
    plt.show()
    plt.close()


def decoder_mask(encoded):
    x = Deconvolution2D(32, (3, 3), input_shape=(48, 4, 4), activation='relu', strides=(2, 2), padding="same")(encoded)  #4*4>>8*8
    x = Dropout(0.25)(x)
    x = Deconvolution2D(32, (3, 3), input_shape=(32, 8, 8),activation='relu', strides=(2, 2), padding="same")(x) #8*8>>16*16
    x = Dropout(0.25)(x)
    #x = BatchNormalization()(x)
    x = Deconvolution2D(16, (3, 3), input_shape=(32, 16, 16),activation='relu', strides=(2, 2), padding='same')(x) #16*16>>32*32
    x = Dropout(0.25)(x)
    x = Deconvolution2D(16, (3, 3), input_shape=(16, 32, 32),activation='relu', strides=(2, 2), padding='same')(x) #32*32>>64*64
    x = Deconvolution2D(16, (3, 3), input_shape=(16, 64, 64),activation='relu', strides=(2, 2), padding='same')(x) #64*64>>128*128
    decoded = Deconvolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


def decoder_mask_test(encoded):
    #
    x = Deconvolution2D(32, (3, 3), input_shape=(48, 4, 4), activation='relu', strides=(2, 2), padding="same")(encoded)  #4*4>>8*8
    x = Deconvolution2D(32, (3, 3), input_shape=(32, 8, 8),activation='relu', strides=(2, 2), padding="same")(x) #8*8>>16*16
    x = Deconvolution2D(16, (3, 3), input_shape=(32, 16, 16),activation='relu', strides=(2, 2), padding='same')(x) #16*16>>32*32
    x = Deconvolution2D(16, (3, 3), input_shape=(16, 32, 32),activation='relu', strides=(2, 2), padding='same')(x) #32*32>>64*64
    x = Deconvolution2D(16, (3, 3), input_shape=(16, 64, 64),activation='relu', strides=(2, 2), padding='same')(x) #64*64>>128*128
    decoded = Deconvolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

def decoder_box(encoded):
    #decoded = Dense(2, activation='relu')(encoded)
    #decoded = Dense(4, activation='relu',input_shape=(4,))(encoded)
    # TODO:
    decoded = Dense(4, activation='sigmoid',input_shape=(128,))(encoded) # dense_4
    return decoded

def scheduler(epoch, lr):
    if epoch < 10:
         return lr
    else:
         return lr * tf.math.exp(-0.1)

def DHAE_Model_reshape(img_y,img_x, n_G, MTL, corr_learn=False):
    #-----------------------------------
    # TODO: configure for multi-gpu
    # TODO: conv layer: n_H, n_W decrease and n_C increase?
    # TODO: learn AE for location and shape feature separately and then use learned weight to learn for MTL
    # TODO: visualize activation
    # TODO: revisit for correcting MTL loss
    #------------------------------------
    input_img = Input(shape=(img_y, img_x, 1))
    input_box = Input(shape=(4,))

    e1 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)  # 128*128>>64*64
    e1 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#64*64>>32*32
    #e1 = BatchNormalization()(e1)
    e1 = Dropout(0.25)(e1)
    e1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#32*32>>16*16
    #e1 = BatchNormalization()(e1)
    e1 = Dropout(0.25)(e1)
    e1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#16*16>>8*8
    #e1 = BatchNormalization()(e1)
    e1 = Dropout(0.25)(e1)
    e1 = Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#8*8>>4*4
    #e1 = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)  #
    #e1 = BatchNormalization()(e1)
    #e1 = Dropout(0.25)(e1)
    e1 = Reshape((768,))(e1)
    e1 = Dense(64,activation='relu',input_shape=(768,))(e1)
    #e1 = BatchNormalization()(e1)
    #e1 = Dropout(0.25)(e1)
    mask_encoder_model = Model(input_img,e1)

    e2 = Dense(64, activation='relu',input_shape=(4,))(input_box) #dense_1
    #e2 = BatchNormalization()(e2)
    box_encoder_model = Model(input_box,e2)
    #concatenated
    concatenated_feature = concatenate([e1,e2])
    concatenated_model = Model([input_img,input_box],concatenated_feature)

    # bottleneck
    bottleneck = Dense(64, activation='relu',input_shape=(128,))(concatenated_feature) # dense_2
    bottleneck_model = Model([input_img,input_box],bottleneck)

    em = Dense(768, activation='relu',input_shape=(64,))(bottleneck)
    eb = Dense(64, activation='relu',input_shape=(64,))(bottleneck)
    em = Reshape((4,4,48))(em)

    decoded_mask = decoder_mask(em)
    decoded_box = decoder_box(eb)
    if MTL:
        #e1 = Softmax()(e1)
        #e2 = Softmax()(e2)
        out = CustomMultiLossLayer(nb_outputs=2)([input_img,input_box,decoded_mask, decoded_box])
        # check to see if we are compiling using just a single GPU
        if n_G <= 1:
            print("[INFO] training with 1 GPU...")
            final_model = Model([input_img, input_box], out)
        # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(n_G))

            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            with tf.device("/cpu:0"):
                # initialize the model
                final_model = Model([input_img, input_box], out)
        #ini_lr = 0.001
        #n_epochs = final_model
        #opt = adadelta(lr=ini_lr,decay=ini_lr/n_epochs)
        #(lr=0.001, decay=.001 / EPOCHS)
        #optimizer_adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #final_model.compile(optimizer= optimizer_adam, loss=None)
        final_model.compile(optimizer='adadelta', loss=None)
    # When arbitrary weights are used for multi-tasking
    if not MTL:
        # check to see if we are compiling using just a single GPU
        if n_G <= 1:
            final_model = Model([input_img, input_box], [decoded_mask,decoded_box])
        else:
            print("[INFO] training with {} GPUs...".format(n_G))
            with tf.device("/cpu:0"):
                final_model = Model([input_img, input_box], [decoded_mask,decoded_box])
        # set loss function for arbitrary weight
        print('Model Training With Arbitrary Multi-Task Weights.....')
        losses_all = {'conv2d_transpose_6': final_loss_mask,'dense_6': final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
        lossWeights = {'conv2d_transpose_6': 0.5, 'dense_6': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
        # opt = adadelta(lr=ini_lr,decay=ini_lr/epochs)
        # (lr=0.001, decay=.001 / EPOCHS)
        #optimizer_adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #final_model.compile(optimizer=optimizer_adam, loss=losses_all,loss_weights=lossWeights)
        final_model.compile(optimizer='adadelta', loss=losses_all, loss_weights=lossWeights)

    print(final_model.summary())
    #assert len(final_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
    #assert len(final_model.losses) == 1
    return final_model, bottleneck_model, box_encoder_model, mask_encoder_model

def test_net(img_y, img_x, n_G, MTL,corr_learn=False):
    #----------------------------------------------------------------------------------------
    # 1. During test time avoid dropout (only need to learn the model faster and reduce overfitting)
    #
    #
    #
    #----------------------------------------------------------------------------------------
    input_img = Input(shape=(img_y, img_x, 1))#layer0
    input_box = Input(shape=(4,))

    e1 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)  # 64 layer1
    e1 = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#>>32 Layer2
    e1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#16
    e1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#8
    e1 = Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same')(e1)#4
    e1 = Reshape((768,))(e1)
    e1 = Dense(64,activation='relu',input_shape=(768,))(e1) #dense_1
    mask_encoder_model = Model(input_img,e1)

    e2 = Dense(64, activation='relu',input_shape=(4,))(input_box) #dense_2
    box_encoder_model = Model(input_box,e2)

    #concatenated
    concatenated_feature = concatenate([e1,e2])
    concatenated_model = Model([input_img,input_box],concatenated_feature)


    # bottleneck
    bottleneck = Dense(64, activation='relu',input_shape=(128,))(concatenated_feature) # dense_3

    bottleneck_model = Model([input_img,input_box],bottleneck)

    em = Dense(768, activation='relu',input_shape=(64,))(bottleneck)

    eb = Dense(64, activation='relu',input_shape=(64,))(bottleneck)
    #em = Dense(768, activation='relu', input_shape=(256,))(em)
    em = Reshape((4,4,48))(em)

    # decode mask and box separately from embedding
    decoded_mask = decoder_mask_test(em)
    decoded_box = decoder_box(eb)
    #out = CustomMultiLossLayer(nb_outputs=2)([ym_true, yb_true, ym_pred, yb_pred])
    if MTL:
        #e1 = Softmax()(e1)
        #e2 = Softmax()(e2)
        out = CustomMultiLossLayer(nb_outputs=2)([input_img,input_box,decoded_mask, decoded_box])
        # check to see if we are compiling using just a single GPU
        if n_G <= 1:
            print("[INFO] training with 1 GPU...")
            final_model = Model([input_img, input_box], out)
        # otherwise, we are compiling using multiple GPUs
        else:
            print("[INFO] training with {} GPUs...".format(n_G))

            # we'll store a copy of the model on *every* GPU and then combine
            # the results from the gradient updates on the CPU
            with tf.device("/cpu:0"):
                # initialize the model
                final_model = Model([input_img, input_box], out)
        optimizer_adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #final_model.compile(optimizer=optimizer_adam, loss=None)
        final_model.compile(optimizer='adadelta', loss=None)
    # When arbitrary weights are used for multi-tasking
    if not MTL:
        # check to see if we are compiling using just a single GPU
        if n_G <= 1:
            final_model = Model([input_img, input_box], [decoded_mask,decoded_box])
        else:
            print("[INFO] training with {} GPUs...".format(n_G))
            with tf.device("/cpu:0"):
                final_model = Model([input_img, input_box], out)
        # set loss function for arbitrary weight
        print('Model Training With Arbitrary Multi-Task Weights.....')
        losses_all = {'conv2d_transpose_6': final_loss_mask,'dense_6': final_loss_box}  # ,'custom_regularization_1':zero_loss 'dense_4':self_express_loss}
        lossWeights = {'conv2d_transpose_6': 0.5, 'dense_6': 0.5}  # ,'custom_regularization_1':0.2 'dense_2': 0.5
        # opt = adadelta(lr=ini_lr,decay=ini_lr/epochs)
        # (lr=0.001, decay=.001 / EPOCHS)
        optimizer_adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #final_model.compile(optimizer=optimizer_adam, loss=losses_all,loss_weights=lossWeights)
        final_model.compile(optimizer='adadelta', loss=losses_all, loss_weights=lossWeights)

    print(final_model.summary())
    #assert len(final_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
    #assert len(final_model.losses) == 1
    return final_model, bottleneck_model, box_encoder_model, mask_encoder_model

def delete_all(demo_path, fmt='ckpt'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)
    else:
        print('{} hase no {} files'.format(demo_path,fmt))
# Train autoencoder - Main Function
if __name__ == '__main__':
    codeBase = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/'
    evaluation = 'train'
    dataset = 'kitti'
    n_G = 1
    MTL = 1
    # Save Checkpoints
    if evaluation == 'train':
        t = time.process_time()
        if dataset=='kitti':
            x_m = np.load(codeBase+'Mask_Instance_Clustering/train_data_all/train_mask_mot_all0_reshape128_kitti.npy',encoding='bytes')
            x_b = np.load(codeBase+'Mask_Instance_Clustering/train_data_all/train_box_mot_all0_reshape128_kitti.npy',encoding='bytes')
            x_train_mask, x_test_mask = prepare_data_mask(x_m)
            x_train_box, x_test_box = prepare_data_box(x_b)
            assert x_train_box.shape[0]==x_train_mask.shape[0],'multi-task instances should be equal but found (mask,box) > ({},{})'.format(x_train_mask.shape[0],x_train_box.shape[0])
            #model_path = codeBase+'Mask_Instance_Clustering/final_model/checkpoints_MOTS_128_kitti_new/'
            #model_path = codeBase+'Mask_Instance_Clustering/final_model/model_exp/arbit_weight/'
            if MTL:
                model_path = codeBase + 'Mask_Instance_Clustering/final_model/model_exp/shape-model-uncorr-loss/MTL/'
            else:
                model_path = codeBase + 'Mask_Instance_Clustering/final_model/model_exp/shape-model/arbit_weight/Kitti_128'

        if dataset=='clasp2':
            x_m = np.load('/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_mask_clasp2_reshape128.npy',encoding='bytes')
            x_b = np.load('/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_box_clasp2_reshape128.npy', encoding='bytes')
            x_train_mask, x_test_mask = prepare_data_mask(x_m)
            x_train_box, x_test_box = prepare_data_box(x_b)
            assert x_train_box.shape[0] == x_train_mask.shape[0], 'multi-task instances should be equal but found (mask,box) > ({},{})'.format(x_train_mask.shape[0],x_train_box.shape[0])
            model_path = '/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/final_model/checkpoints_clasp2_adam/'

        '''
        aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
        '''
        img_y=128
        img_x=128
        # clear old files
        delete_all(model_path, fmt='ckpt')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint_path = model_path + 'cp-{epoch:04d}.ckpt'#'final_model.cpt'#'cp-{epoch:04d}.ckpt'
        #checkpoint_path = model_path + 'cp-{epoch:04d}.hdf5'
        checkpoint_dir =os.path.dirname(checkpoint_path)
        #cp_callback = callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_best_only=True,save_weights_only=True,period=10,)
        cp_callback = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True, mode='min')
        #cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',save_weights_only=True, verbose=1, save_best_only=True, mode='min')
        #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        #cp_callback_stop = callbacks.EarlyStopping(monitor=['val_loss','train_loss'],min_delta=0.00001,patience=100,verbose=1,restore_best_weights=True,mode='min')
        #final_model.save_weights(checkpoint_path.format(epoch=0))
        #final_model.load_weights('/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weightv1/cp-0500.ckpt')
        final_model, bottleneck_model, box_encoder_model, mask_encoder_model = DHAE_Model_reshape(img_y,img_x, n_G, MTL)
        if MTL:
            final_model.fit([x_train_mask,x_train_box], # best parameter = [300,ThaHi 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=1024,
                            shuffle=True,
                            validation_split=0.1,
                            )
        if not MTL:
            final_model.fit([x_train_mask,x_train_box],[x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=1024,
                            shuffle=True,
                            validation_split=0.1
                            )
        '''
        # train the network
        final_model.fit([x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                        epochs=500,
                        callbacks=[cp_callback],
                        #batch_size=256 * n_G,
                        steps_per_epoch=len(x_train_mask) // (512 * n_G),
                        shuffle=True,
                        validation_split=0.1,
                        validation_steps = len(x_train_mask) // (512 * n_G)
                        )
        '''
        if MTL:
            input_img, input_box, decoded_imgs, decoded_box = final_model.predict([x_test_mask, x_test_box])
        else:
            decoded_imgs,decoded_box = final_model.predict([x_test_mask,x_test_box])
        #concat_feature = concatenated_model.predict([x_test_mask,x_test_box])

        bottleneck_feature = bottleneck_model.predict([x_test_mask,x_test_box])
        import matplotlib.pylab as pylab
        pylab.plot(final_model.history.history['loss'])
        pylab.show()
        loss_curve(final_model,model_path)
        visualize_reconstruction(x_test_mask, bottleneck_feature, decoded_imgs,img_y,img_x,model_path)
        if MTL:
            # Found Standard Deviations
            #print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in final_model.layers[-1].log_vars])
            print([K.get_value(log_var[0]) ** 0.5 for log_var in final_model.layers[-1].log_vars])

        print('%f' % (time.process_time()-t))

    if evaluation == 'train_2nd_phase':
        t = time.process_time()
        #x_m = np.load('/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/train_data_all/train_mask_mot_all0_reshape128_kitti.npy',encoding='bytes')
        x_m = np.load(
            '/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_mask_clasp2_reshape128.npy',
            encoding='bytes')
        x_train_mask,x_test_mask = prepare_data_mask(x_m)

        #x_b = np.load('/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/train_data_all/train_box_mot_all0_reshape128_kitti.npy',encoding='bytes')
        x_b = np.load(
             '/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_box_clasp2_reshape128.npy',
            encoding='bytes')
        x_train_box, x_test_box = prepare_data_box(x_b)
        '''
        aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
        '''

        img_y=128
        img_x=128
        model_path = '/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/final_model/checkpoints_clasp2_adam/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint_path = model_path + 'cp-{epoch:04d}.ckpt'#'final_model.cpt'#'cp-{epoch:04d}.ckpt'
        checkpoint_dir =os.path.dirname(checkpoint_path)
        #cp_callback = callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_best_only=True,save_weights_only=True,period=10,)

        cp_callback = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        #cp_callback_stop = callbacks.EarlyStopping(monitor=['val_loss','train_loss'],min_delta=0.00001,patience=100,verbose=1,restore_best_weights=True,mode='min')
        #final_model.save_weights(checkpoint_path.format(epoch=0))
        #final_model.load_weights('/media/siddique/Data/cae_trained_model_mask/final_model/checkpoints/loss-weightv1/cp-0500.ckpt')
        final_model, bottleneck_model, box_encoder_model, mask_encoder_model = DHAE_Model_reshape(img_y,img_x, n_G, MTL)

        if MTL:
            checkpoint = '/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/final_model/checkpoints_clasp2_adam/cp-0249.ckpt'
            final_model.load_weights(checkpoint)
            input_img,input_box,decoded_mask, decoded_box = final_model.predict([x_test_mask, x_test_box])
            # concat_feature = concatenated_model.predict([x_test_mask,x_test_box])

            bottleneck_feature = bottleneck_model.predict([x_test_mask, x_test_box])
            import matplotlib.pylab as pylab

            #pylab.plot(final_model.history.history['loss'])
            #pylab.show()
            #loss_curve(final_model, model_path)
            visualize_reconstruction(x_test_mask, bottleneck_feature, decoded_mask, img_y, img_x, model_path)

            # Found Standard Deviations
            print([np.exp(K.get_value(log_var[0])) ** 0.5 for log_var in final_model.layers[-1].log_vars])

            print('%f' % (time.process_time() - t))

            final_model.fit([x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=512,
                            shuffle=True,
                            validation_split=0.1,
                            )
        if not MTL:
            final_model.fit([x_train_mask,x_train_box],[x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                            epochs=500,
                            callbacks=[cp_callback],
                            batch_size=512,
                            shuffle=True,
                            validation_split=0.1
                            )
        '''
        # train the network
        final_model.fit([x_train_mask,x_train_box], # best parameter = [300, 512, 0.1]
                        epochs=500,
                        callbacks=[cp_callback],
                        #batch_size=256 * n_G,
                        steps_per_epoch=len(x_train_mask) // (512 * n_G),
                        shuffle=True,
                        validation_split=0.1,
                        validation_steps = len(x_train_mask) // (512 * n_G)
                        )
        '''

