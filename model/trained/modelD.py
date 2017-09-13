from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

num_features_mul = 1

def down_layer(input, n_features):
    down = Conv2D(n_features, (3, 3), padding='same')(input)
    # down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Conv2D(n_features, (3, 3), padding='same')(down)
    # down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)
    return down, down_pool

def up_layer(input, down, n_features):
    up = UpSampling2D((2, 2))(input)
    up = concatenate([down, up], axis=3)
    up = Conv2D(n_features, (3, 3), padding='same')(up)
    # up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv2D(n_features, (3, 3), padding='same')(up)
    # up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv2D(n_features, (3, 3), padding='same')(up)
    # up = BatchNormalization()(up)
    up = Activation('relu')(up)
    return up

def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    down0b, down0b_pool = down_layer(inputs, 32 * num_features_mul)
    # 512

    down0a, down0a_pool = down_layer(down0b_pool, 32 * num_features_mul)
    # 256

    down0, down0_pool = down_layer(down0a_pool, 32 * num_features_mul)
    # 128

    down1, down1_pool = down_layer(down0_pool, 64 * num_features_mul)
    # 64

    down2, down2_pool = down_layer(down1_pool, 128 * num_features_mul)
    # 32

    down3, down3_pool = down_layer(down2_pool, 256 * num_features_mul)
    # 16

    down4, down4_pool = down_layer(down3_pool, 512 * num_features_mul)
    # 8

    center = Conv2D(1024 * num_features_mul, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024 * num_features_mul, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = up_layer(center, down4, 512 * num_features_mul)
    # 16

    up3 = up_layer(up4, down3, 256 * num_features_mul)
    # 32

    up2 = up_layer(up3, down2, 128 * num_features_mul)
    # 64

    up1 = up_layer(up2, down1, 64 * num_features_mul)
    # 128

    up0 = up_layer(up1, down0, 32 * num_features_mul)
    # 256

    up0a = up_layer(up0, down0a, 32 * num_features_mul)
    # 512

    up0b = up_layer(up0a, down0b, 32 * num_features_mul)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    # model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_bce_dice_loss, metrics=[dice_coeff])

    return model