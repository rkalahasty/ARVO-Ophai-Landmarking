
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


def down_block(x, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x

def up_block(x, filters, concat, dropout):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([x, concat])
    x = Dropout(dropout)(x)
    x = down_block(x, filters)
    return x


def Unet(input_size=(256, 256, 3), n_classes=2, n_filters=64, dropout=0.1):
    inputs = tf.keras.Input(shape=input_size)
    c1 = down_block(inputs, n_filters)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = down_block(p1, n_filters * 2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = down_block(p2, n_filters * 4)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    # c4 = down_block(p3, n_filters * 8)
    # p4 = MaxPooling2D((2, 2))(c4)
    # p4 = Dropout(dropout)(p4)

    # c5 = down_block(p4, n_filters * 16)
    c4 = down_block(p3, n_filters * 8)

    # c6 = up_block(c5, n_filters * 8, c4, dropout)
    c7 = up_block(c4, n_filters * 4, c3, dropout)
    c8 = up_block(c7, n_filters * 2, c2, dropout)
    c9 = up_block(c8, n_filters, c1, dropout)

    outputs = Conv2D(2, 1, activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)
    return model
