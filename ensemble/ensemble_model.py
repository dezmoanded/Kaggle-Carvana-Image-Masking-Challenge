from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

batch_size = 12

def get_ensemble_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(16, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Activation('relu')(conv2)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv1)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.001), loss=weighted_bce_dice_loss, metrics=[dice_coeff])

    return model