from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
import tensorflow as tf

import numpy as np
import json
import os
import pickle

from Models.VAE_Keras import DenseVariationalAutoencoderKeras

class EntanglementSoftmax(DenseVariationalAutoencoderKeras):
  def __init__(self, input_dim
        , encoder_dense_units
        , decoder_dense_units
        , z_dim
        , use_batch_norm = False
        , use_dropout= False
        ):
        super().__init__(input_dim
                        , encoder_dense_units
                        , decoder_dense_units
                        , z_dim
                        , use_batch_norm = use_batch_norm
                        , use_dropout= use_dropout
                        )

  def _build(self):
        
        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        
        x = encoder_input

        for i in range(self.n_layers_encoder):
            
            dense_layer = Dense( 
                self.encoder_dense_units[i]
                , name = 'encoder_dense_' + str(i)
                )
            
            x = dense_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
      
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)
                
        ### THE DECODER

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):            
            
            dense_layer = Dense( 
                self.decoder_dense_units[i]
                , name = 'decoder_dense_' + str(i)
                )

            x = dense_layer(x)

            if i < self.n_layers_decoder - 1: 
                
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = LeakyReLU()(x)
                
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            
            else:
                #x = Activation('linear')(x)
                x = Activation('softmax')(x)
            

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
    
  def compile(self, learning_rate, r_loss_factor, Beta):
      """
      Compiling the network. Need to choose the learning rate, r_loss_factor
      and Beta for the Beta-VAE, if Beta = 1 then it is a VAE.
        
      Parameters
      ----------------------------------------------------------------------
      learning_rate: Learning Rate for gradient descent.
      r_loss_factor: Factor that multiplies the loss factor of the
                     reconstruction loss.
      Beta: Beta-VAE parameter that multiplies the KL-Divergence in order to
            disentangle the latent space of the model.
              
      """
        
      self.learning_rate = learning_rate        

      ### COMPILATION
      def vae_r_loss(y_true, y_pred):
          r_loss = categorical_crossentropy(y_true, y_pred)
          return r_loss_factor * r_loss

      def vae_kl_loss(y_true, y_pred):
          kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
          return kl_loss

      def vae_loss(y_true, y_pred):
          r_loss = vae_r_loss(y_true, y_pred)
          kl_loss = vae_kl_loss(y_true, y_pred)
          return  r_loss + Beta*kl_loss

      optimizer = Adam(lr=learning_rate)
      self.model.compile(optimizer=optimizer, 
                         loss = vae_loss,  
                         metrics = ['accuracy', 
                                    vae_r_loss, 
                                    vae_kl_loss]
                        )

        