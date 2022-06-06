# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 02:28:23 2020
@author: edward
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, \
    Conv3D, Conv3DTranspose, UpSampling3D, \
    MaxPooling3D, Dropout
import numpy as np
"""
Includes implementation for Bayesian 3D U-Net model
Assumes input is of shape: (batch x frames x h x w x channels)
"""
class BatchNormRelu3D(tf.keras.layers.Layer):
    """Batch normalization + ReLu"""
    def __init__(self, batch_normalization=True, instance_normalization=False,
                 name=None, dtype=None):
        super(BatchNormRelu3D, self).__init__(name=name)
        self._batch_normalization = batch_normalization
        self._instance_normalization = instance_normalization
        if batch_normalization:
            self.norm = BatchNormalization(axis=-1, dtype=dtype)
        elif instance_normalization:
            raise NotImplementedError
        self.relu = tf.keras.layers.ReLU(dtype=dtype)
    
    def call(self, inputs, is_training):
        x = inputs
        if self._batch_normalization:
            x = self.norm(x, training=is_training)
        x = self.relu(x)
        return x
    
class DownConvBlock3D(tf.keras.layers.Layer):
    """
    Downsampling ConvBlock3D on Encoder side
    Assumes input is of shape: (batch x frames x h x w x channels)
    """
    def __init__(self, filters, kernel_size=(3,3,3),
                 strides=(1,1,1), padding='same', pool_size=(2,2,2),
                 do_max_pool=True, regularizer=None, dropout=0.25, dropout_type='block', 
                 batch_norm=True, name=None):
        """
        dropout_type: 'all', 'block', or 'none'
            - 'all': dropout after each conv layer
            - 'block': dropout only after entire block
            - 'none': no dropout
        """
        super(DownConvBlock3D, self).__init__(name=name)
        self.do_max_pool = do_max_pool
        self.dropout_type = dropout_type
        self.conv1 = Conv3D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_regularizer=regularizer)
        self.brelu1 = BatchNormRelu3D(batch_normalization=batch_norm,
                                      instance_normalization=False,
                                      )
        self.conv2 = Conv3D(filters=filters*2,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_regularizer=regularizer)
        self.brelu2 = BatchNormRelu3D(batch_normalization=batch_norm,
                                      instance_normalization=False
                                      )
        self.max_pool = MaxPooling3D(pool_size=pool_size)
        self.dropout1 = Dropout(rate=dropout)
        self.dropout2 = Dropout(rate=dropout)
        self.dropoutb = Dropout(rate=dropout)
    
    def call(self, inputs, is_training, dropout_training):
        x = self.conv1(inputs)
        if self.dropout_type == 'all':
            x = self.dropout1(x, training=dropout_training)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        if self.dropout_type == 'all':
            x = self.dropout2(x, training=dropout_training)
        x = self.brelu2(x, is_training)
        if self.dropout_type == 'block':
            x = self.dropoutb(x, training=dropout_training)
        skip_output = x
        if self.do_max_pool:
            x = self.max_pool(x)
        return x, skip_output
    
class UpConvBlock3D(tf.keras.layers.Layer):
    """
    Upsampling ConvBlock3D on Decoder side
    Assumes input is of shape: (batch x frames x h x w x channels)
    """
    def __init__(self, up_filters, reg_filters, up_kernel_size=(2,2,2),
                 reg_kernel_size=(3,3,3), padding='same', pool_size=(2,2,2),
                 transposed_conv=True, dropout=0.25, regularizer=None, dropout_type='block',
                 batch_norm=True, name=None):
        """
        dropout_type: 'all', 'block', or 'none'
            - 'all': dropout after each conv layer
            - 'block': dropout only after entire block
            - 'none': no dropout
        """
        super(UpConvBlock3D, self).__init__(name=name)
        self.dropout_type = dropout_type
        if transposed_conv:
            self.up_conv = Conv3DTranspose(filters=up_filters,
                                           kernel_size=up_kernel_size,
                                           strides=(2,2,2),
                                           kernel_regularizer=regularizer)
        else:
            self.up_conv = UpSampling3D(size=pool_size)
        self.conv1 = Conv3D(filters=reg_filters,
                           kernel_size=reg_kernel_size,
                           strides=(1,1,1),
                           padding=padding,
                           kernel_regularizer=regularizer)
        self.brelu1 = BatchNormRelu3D(batch_normalization=batch_norm,
                                      instance_normalization=False,
                                      )
        self.conv2 = Conv3D(filters=reg_filters,
                            kernel_size=reg_kernel_size,
                            strides=(1,1,1),
                            padding=padding,
                            kernel_regularizer=regularizer)
        self.brelu2 = BatchNormRelu3D(batch_normalization=batch_norm,
                                      instance_normalization=False
                                      )
        self.dropout1 = Dropout(rate=dropout)
        self.dropout2 = Dropout(rate=dropout)
        self.dropoutb = Dropout(rate=dropout)
    
    def call(self, inputs, skip_inputs, is_training, dropout_training):
        x = self.up_conv(inputs)
        x = tf.concat([x, skip_inputs], axis=-1)
        x = self.conv1(x)
        if self.dropout_type == 'all':
            x = self.dropout1(x, training=dropout_training)
        x = self.brelu1(x, is_training)
        x = self.conv2(x)
        if self.dropout_type == 'all':
            x = self.dropout2(x, training=dropout_training)
        x = self.brelu2(x, is_training)
        if self.dropout_type == 'block':
            x = self.dropoutb(x, training=dropout_training)
        return x
                
class BayesUNet3DKG(tf.keras.Model):
    """
    Overall 3D U-Net Architecture
    Assumes input is of shape: (batch x frames x h x w x channels)
    Due to the max pooling layer, the depth of the Bayesian U-Net
    should satisfy: 2^(depth) < frames
    """
    def __init__(self, 
                 num_classes, 
                 depth=4,
                 dropout=0.25,
                 batch_norm=True,
                 regularizer=None,
                 dropout_type='block',
                 data_format='channels_last'):
        super(BayesUNet3DKG, self).__init__()
        assert depth > 2, "Depth should at least be greater than 2"
        self.base_num_filters = 32
        self.num_classes = num_classes
        self.eblocks_list = []
        for layer_depth in range(depth):
            self.eblocks_list.append(
                                    DownConvBlock3D(
                                        filters=self.base_num_filters*(2**layer_depth),
                                        name='eblock_{}'.format(layer_depth+1),
                                        do_max_pool=(False if layer_depth==depth-1 else True),
                                        regularizer=regularizer,
                                        dropout=dropout,
                                        dropout_type=dropout_type,
                                        batch_norm=batch_norm
                                                )
                                    )
        layer_depth = depth-2
        self.dblock_shared = UpConvBlock3D(
                                        up_filters=self.base_num_filters*(2**(layer_depth+1)),
                                        reg_filters=self.base_num_filters*(2**layer_depth),
                                        name='dblock_{}'.format(layer_depth+1),
                                        dropout=dropout,
                                        regularizer=regularizer,
                                        dropout_type=dropout_type,
                                        batch_norm=batch_norm
                                                  )
        self.dblocks_list_mean = []
        for layer_depth in range(depth-3, -1, -1):
            self.dblocks_list_mean.append(
                                    UpConvBlock3D(
                                        up_filters=self.base_num_filters*(2**(layer_depth+1)),
                                        reg_filters=self.base_num_filters*(2**layer_depth),
                                        name='dblock_mean_{}'.format(layer_depth+1),
                                        dropout=dropout,
                                        regularizer=regularizer,
                                        dropout_type='none',
                                        batch_norm=batch_norm
                                                  )
                                    )
        self.dblocks_list_logvar = []
        for layer_depth in range(depth-3, -1, -1):
            self.dblocks_list_logvar.append(
                                    UpConvBlock3D(
                                        up_filters=self.base_num_filters*(2**(layer_depth+1)),
                                        reg_filters=self.base_num_filters*(2**layer_depth),
                                        name='dblock_logvar_{}'.format(layer_depth+1),
                                        dropout=dropout,
                                        regularizer=regularizer,
                                        dropout_type='none', 
                                        batch_norm=batch_norm
                                                  )
                                    )
        self.conv_final_mean = Conv3D(filters=num_classes,
                                 kernel_size=(1,1,1),
                                 name='conv_final_mean'
                                 )
        self.conv_final_logvar = Conv3D(filters=num_classes,
                                 kernel_size=(1,1,1),
                                 name='conv_final_logvar'
                                 )
        
    # @tf.function
    def call(self, inputs, is_training=True, dropout_training=True):
        """
        Assumes input is of shape: (batch x frames x h x w x channels)
        is_training: flag for batch normalization training functionality
        dropout_training: flag for dropout layers training functionality
        """
        # print("Calling model!")
        x = inputs
        skip_outputs = []
        for eblock in self.eblocks_list:
            x, skip_output = eblock(x, is_training, 
                                    dropout_training=dropout_training)
            skip_outputs.append(skip_output)
            
        skip_outputs = skip_outputs[::-1]
        x = self.dblock_shared(x, skip_outputs[1], is_training,
                               dropout_training=dropout_training)
        
        mean, logvar = x, x
        for d, dblock in enumerate(self.dblocks_list_mean):
            mean = dblock(mean, skip_outputs[d+2], is_training,
                          dropout_training=dropout_training)
        outputs_mean = self.conv_final_mean(mean)
        
        for d, dblock in enumerate(self.dblocks_list_logvar):
            logvar = dblock(logvar, skip_outputs[d+2], is_training,
                            dropout_training=dropout_training)
        outputs_logvar = self.conv_final_logvar(logvar)
        
        d = outputs_mean.get_shape().as_list()[1] # (SHOULD BE FRAMESx256x256x3 if classes is 3)
        h = outputs_mean.get_shape().as_list()[2]
        w = outputs_mean.get_shape().as_list()[3]
        outputs_reshaped_mean = tf.reshape(outputs_mean, np.asarray([-1, self.num_classes]))
        outputs_softmax_pre_shape_mean = tf.keras.activations.softmax(outputs_reshaped_mean)
        outputs_logits_pre_shape_mean = outputs_reshaped_mean
        outputs_logits_mean = tf.reshape(outputs_logits_pre_shape_mean, np.asarray([-1, d, h, w, self.num_classes]))
        outputs_softmax_mean = tf.reshape(outputs_softmax_pre_shape_mean, np.asarray([-1, d, h, w, self.num_classes]))
        
        d = outputs_logvar.get_shape().as_list()[1] # (SHOULD BE FRAMESx256x256x3 if classes is 3)
        h = outputs_logvar.get_shape().as_list()[2]
        w = outputs_logvar.get_shape().as_list()[3]
        outputs_reshaped_logvar = tf.reshape(outputs_logvar, np.asarray([-1, self.num_classes]))
        outputs_softmax_pre_shape_logvar = tf.keras.activations.softmax(outputs_reshaped_logvar)
        outputs_logits_pre_shape_logvar = outputs_reshaped_logvar
        outputs_logits_logvar = tf.reshape(outputs_logits_pre_shape_logvar, np.asarray([-1, d, h, w, self.num_classes]))
        outputs_softmax_logvar = tf.reshape(outputs_softmax_pre_shape_logvar, np.asarray([-1, d, h, w, self.num_classes]))
        
        return outputs_logits_mean, outputs_softmax_mean, outputs_logits_logvar, outputs_softmax_logvar