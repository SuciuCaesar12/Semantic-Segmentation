from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Concatenate, Conv2DTranspose, BatchNormalization, Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model
import tensorflow as tf
from utils import read_yaml_file
import os


class ContractingBlock(Layer):

    def __init__(self, dict_block):
        super(ContractingBlock, self).__init__()

        self.batch_norm_flag = dict_block['batch_norm']

        self.conv_1 = Conv2D(filters=dict_block['filters'],
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             activation=dict_block['activation'],
                             kernel_initializer=dict_block['kernel_initializer'],
                             kernel_regularizer=L2(dict_block['regularizer_factor']))
        self.conv_2 = Conv2D(filters=dict_block['filters'],
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             activation=dict_block['activation'],
                             kernel_initializer=dict_block['kernel_initializer'],
                             kernel_regularizer=L2(dict_block['regularizer_factor']))
        if self.batch_norm_flag:
            self.batch_norm_1 = BatchNormalization()
            self.batch_norm_2 = BatchNormalization()
        self.maxpool = MaxPool2D(pool_size=(2, 2),
                                 strides=2)

    def call(self, x_input):
        x = self.conv_1(x_input)
        x = self.batch_norm_1(x) if self.batch_norm_flag else x
        x = self.conv_2(x)
        x = self.batch_norm_2(x) if self.batch_norm_flag else x
        return self.maxpool(x), x


class ExpandingBlock(Layer):

    def __init__(self, dict_block, output_block=False):
        super(ExpandingBlock, self).__init__()

        self.batch_norm_flag = dict_block['batch_norm']
        self.output_block = output_block

        self.concat_layer = Concatenate()
        self.conv_1 = Conv2D(filters=dict_block['filters'],
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             activation=dict_block['activation'],
                             kernel_initializer=dict_block['kernel_initializer'],
                             kernel_regularizer=L2(dict_block['regularizer_factor']))
        self.conv_2 = Conv2D(filters=dict_block['filters'],
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             activation=dict_block['activation'],
                             kernel_initializer=dict_block['kernel_initializer'],
                             kernel_regularizer=L2(dict_block['regularizer_factor']))
        if self.batch_norm_flag:
            self.batch_norm_1 = BatchNormalization()
            self.batch_norm_2 = BatchNormalization()

        self.up_conv = Conv2DTranspose(filters=dict_block['filters'] // 2,
                                       kernel_size=(1, 1),
                                       padding='valid',
                                       strides=2,
                                       activation=dict_block['activation'],
                                       kernel_initializer=dict_block['kernel_initializer'],
                                       kernel_regularizer=L2(dict_block['regularizer_factor']))

    def call(self, x_input, x_skip):
        x = self.concat_layer([x_skip, x_input]) if x_skip is not None else x_input
        x = self.conv_1(x)
        x = self.batch_norm_1(x) if self.batch_norm_flag else x
        x = self.conv_2(x)
        x = self.batch_norm_2(x) if self.batch_norm_flag else x
        x = self.up_conv(x) if not self.output_block else x
        return x


class Unet:

    def __init__(self, model_info):
        self.INPUT_SHAPE = eval(model_info['INPUT_SHAPE'])
        self.C = model_info['C']  # nb of classes
        self.n_blocks = model_info['n_blocks']

        self.activation = model_info['activation']
        self.kernel_initializer = model_info['kernel_initializer']
        self.regularizer_factor = model_info['regularizer_factor']
        self.batch_norm_flag = model_info['batch_normalization']

        self.model = self.__create_model()

        if model_info['summary']:
            self.model.summary()
        if model_info['plot_model']:
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            tf.keras.utils.plot_model(self.model, to_file='unet.png',
                                      show_shapes=True,
                                      show_layer_names=True)

    def __contracting_path(self, x, dict_block):
        skip_connections = {}
        for i in range(self.n_blocks):
            x, x_skip = ContractingBlock(dict_block=dict_block)(x)
            dict_block['filters'] *= 2
            skip_connections[str(i)] = x_skip
        return x, skip_connections

    def __expanding_path(self, x, skip_connections, dict_block):
        for i in range(self.n_blocks):
            dict_block['filters'] /= 2
            x = ExpandingBlock(dict_block=dict_block,
                               output_block=(i == self.n_blocks - 1))(x_input=x, x_skip=skip_connections[str(self.n_blocks - 1 - i)])
        return x

    def __bottleneck(self, x_input, dict_block):
        x = Conv2D(filters=dict_block['filters'],
                   kernel_size=(3, 3),
                   padding='same',
                   strides=1,
                   activation=dict_block['activation'],
                   kernel_initializer=dict_block['kernel_initializer'],
                   kernel_regularizer=L2(dict_block['regularizer_factor']))(x_input)
        x = Conv2D(filters=dict_block['filters'],
                   kernel_size=(3, 3),
                   padding='same',
                   strides=1,
                   activation=dict_block['activation'],
                   kernel_initializer=dict_block['kernel_initializer'],
                   kernel_regularizer=L2(dict_block['regularizer_factor']))(x)
        x = Conv2DTranspose(filters=dict_block['filters'] // 2,
                            kernel_size=(1, 1),
                            padding='valid',
                            strides=2,
                            activation=dict_block['activation'],
                            kernel_initializer=dict_block['kernel_initializer'],
                            kernel_regularizer=L2(dict_block['regularizer_factor']))(x)
        return x

    def __create_model(self):
        dict_block = {'filters': 64,
                      'activation': self.activation,
                      'kernel_initializer': self.kernel_initializer,
                      'regularizer_factor': self.regularizer_factor,
                      'batch_norm': self.batch_norm_flag
                      }
        # CONTRACTING PATH ------------------------------------------------------------------------------------------- #
        inputs = Input(shape=self.INPUT_SHAPE, name='Inputs')
        x, skip_connections = self.__contracting_path(x=inputs,
                                                      dict_block=dict_block)
        # BOTTLENECK ------------------------------------------------------------------------------------------------- #
        x = self.__bottleneck(x_input=x,
                              dict_block=dict_block)
        # EXPANDING PATH --------------------------------------------------------------------------------------------- #
        x = self.__expanding_path(x=x,
                                  skip_connections=skip_connections,
                                  dict_block=dict_block)
        x = Conv2D(filters=self.C,
                   kernel_size=(1, 1),
                   padding='same',
                   strides=1,
                   activation=dict_block['activation'],
                   kernel_initializer=dict_block['kernel_initializer'],
                   kernel_regularizer=L2(dict_block['regularizer_factor']))(x)
        outputs = softmax(x)

        return Model(inputs, outputs, name='unet')


if __name__ == '__main__':
    unet = Unet(model_info=read_yaml_file(path='../config.yaml')['unet_info'])
