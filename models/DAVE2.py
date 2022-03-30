import numpy as np
import cv2
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
import os
from PIL import Image
import PIL

class DAVE2Model:

    def __init__(self):
        self.model = Sequential()
        self.input_shape = (150, 200, 3) #(960,1280,3)

    def define_model(self):
        # Start of MODEL Definition
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(1, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # filename = '{}/model.h5'.format(dir_path)
        # self.model.load_weights(filename)
        return self.model

    def define_model_BeamNG(self, h5_filename):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(1, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        self.load_weights(h5_filename)
        return self.model

    # outputs vector [steering, throttle]
    def define_dual_model_BeamNG(self):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(2, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        return self.model

    def foo(self, ip):
        # print(len(ip))
        # a = ip[1]
        # x = ip[0]
        # b = ip[2]
        # return a * x + b
        return ip[0] / 127.5 - 1.

    def define_multi_input_model_BeamNG(self, inputshape=(150, 200, 3)):
        self.input_shape = inputshape
        # # Start of DAVE2 model Definition
        speed_input = Input(shape=(1,), name='speed')
        image_input = Input(shape=(150, 200, 3), name='image')
        # x = Dense(32, activation="relu", input_dim=(150, 200, 3))(image_input)
        # x = Lambda(self.foo)(image_input)  # Important: You can give list of inputs to Lambda layer
        x = Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm')(image_input)

        # 5x5 Convolutional layers with stride of 2x2
        x = Conv2D(24, 5, 2, name='conv1')(x)
        x = ELU(name='elu1')(x)
        x = Conv2D(36, 5, 2, name='conv2')(x)
        x = ELU(name='elu2')(x)
        x = Conv2D(48, 5, 2, name='conv3')(x)
        x = ELU(name='elu3')(x)

        # # 3x3 Convolutional layers with stride of 1x1
        x = Conv2D(64, 3, 1, name='conv4')(x)
        x = ELU(name='elu4')(x)
        x = Conv2D(64, 3, 1, name='conv5')(x)
        x = ELU(name='elu5')(x)

        # Flatten before passing to the fully connected layers
        x = Flatten()(x)

        xspeed = Dense(8, activation="relu")(speed_input)
        xspeed = Dense(8, activation="sigmoid")(xspeed)
        combined = tf.keras.layers.Concatenate()([x, xspeed]) # this works
        x = Dense(100, name='fc0')(combined)

        # Three fully connected layers
        x = Dense(100, name='fc1')(x)
        x = Dropout(.5, name='do1')(x)
        x = ELU(name='elu6')(x)
        x = Dense(50, name='fc2')(x)
        x = Dropout(.5, name='do2')(x)
        x = ELU(name='elu7')(x)
        x = Dense(10, name='fc3')(x)
        x = Dropout(.5, name='do3')(x)
        x = ELU(name='elu8')(x)

        # Output layer with tanh activation
        x = Dense(2, activation='tanh', name='output')(x)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model = Model(inputs=[image_input, speed_input], outputs=x)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        self.model.summary()
        return self.model

    # outputs vector [steering, throttle]
    def define_multi_input_model_BeamNG3(self, inputshape):
        # Start of MODEL Definition
        speed_input = Input(shape=(1,), name='speed')
        image_input = Input(shape=(150, 200, 3), name='image')
        x = Dense(32, activation="relu", input_dim=(150, 200, 3))(image_input)
        x = Lambda(self.foo)([x, speed_input])  # Important: You can give list of inputs to Lambda layer

        # 5x5 Convolutional layers with stride of 2x2
        x = Conv2D(24, 5, 2, name='conv1')(x)
        x = ELU(name='elu1')(x)
        x = Conv2D(36, 5, 2, name='conv2')(x)
        x = ELU(name='elu2')(x)
        x = Conv2D(48, 5, 2, name='conv3')(x)
        x = ELU(name='elu3')(x)

        # Output layer with tanh activation
        x = Dense(2, activation='tanh', name='output')(x)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model = Model(inputs=[image_input, speed_input], outputs=x)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        return self.model

    # outputs vector [steering, throttle]
    def define_multi_input_model_BeamNG2(self, inputshape):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        # self.input_shape = inputs
        a = Input(shape=(1,), name='speed')
        b = Input(shape=(1,), name='extra')
        ip = Input(shape=(784,), name='image')
        x = Dense(32, activation="relu", input_dim=784)(ip)
        x = Lambda(self.foo)([x, a, b])  # Important: You can give list of inputs to Lambda layer
        x = Dense(10, activation="softmax")(x)
        model = Model(inputs=[ip, a, b], outputs=x)
        self.model.add(x)
        # self.model.add(Lambda(lambda x: x / 127.5 - 1., self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(2, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        return self.model

    def atan_layer(self, x):
        return tf.multiply(tf.atan(x), 2)

    def atan_layer_shape(self, input_shape):
        return input_shape

    def define_model_DAVEorig(self):
        input_tensor = Input(shape=(100, 100, 3))
        model = Sequential()
        model.add(Conv2D(24, 5, 2, padding='valid', activation='relu', name='block1_conv1'))
        model.add(Conv2D(36, 5, 2, padding='valid', activation='relu',  name='block1_conv2'))
        model.add(Conv2D(48, 5, 2, padding='valid', activation='relu', name='block1_conv3'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv4'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv5'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(1164, activation='relu', name='fc1'))
        model.add(Dense(100, activation='relu', name='fc2'))
        model.add(Dense(50, activation='relu', name='fc3'))
        model.add(Dense(10, activation='relu', name='fc4'))
        model.add(Dense(1, name='before_prediction'))
        model.add(Lambda(lambda x: tf.multiply(tf.atan(x), 2), output_shape=input_tensor, name='prediction'))
        model.compile(loss='mse', optimizer='adadelta')
        self.model = model
        return model

    def load_weights(self, h5_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = '{}/{}'.format(dir_path, h5_file)
        self.model.load_weights(filename)
        return self.model

    @classmethod
    def process_image(cls, image):
        # image = image.crop((0, 200, 512, 369))
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        image = cv2.resize(image, (200,150))
        image = np.array(image).reshape(1,150,200,3)
        return image

    # Functions to read and preprocess images
    def readProcess(self, image_file):
        """Function to read an image file and crop and resize it for input layer

        Args:
          image_file (str): Image filename (expected in 'data/' subdirectory)

        Returns:
          numpy array of size 66x200x3, for the image that was read from disk
        """
        # Read file from disk
        image = mpimg.imread('data/' + image_file.strip())
        # Remove the top 20 and bottom 20 pixels of 160x320x3 images
        image = image[20:140, :, :]
        # Resize the image to match input layer of the model
        resize = (self.input_shape[0], self.input_shape[1])
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        return image

    def randBright(self, image, br=0.25):
        """Function to randomly change the brightness of an image

        Args:
          image (numpy array): RGB array of input image
          br (float): V-channel will be scaled by a random between br to 1+br

        Returns:
          numpy array of brighness adjusted RGB image of same size as input
        """
        rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rand_bright = br + np.random.uniform()
        rand_image[:,:,2] = rand_image[:,:,2]*rand_bright
        rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
        return rand_image
