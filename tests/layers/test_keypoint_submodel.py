import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout, BatchNormalization, Activation, Conv2D
from keras.utils import plot_model


options = {
    'kernel_size'        : 3,
    'strides'            : 1,
    'padding'            : 'same',
    'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    'bias_initializer'   : 'zeros'
}

inputs = Input(shape=(14, 14, 256))
outputs = inputs
for i in range(4):
    outputs = keras.layers.Conv2D(
        filters=512,
        activation='relu',
        **options
    )(outputs)

outputs = keras.layers.Conv2DTranspose(512, (2,2), activation='relu', strides=2, padding='same')(outputs)
outputs = keras.layers.UpSampling2D(size=(2,2))(outputs)
outputs = keras.layers.Conv2D(9*17, (1,1), activation='softmax', padding='same')(outputs)

model = Model(inputs=inputs, outputs=outputs)

plot_model(model, show_shapes=True)