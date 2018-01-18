from keras.layers.pooling import MaxPooling2D
from keras.initializers import random_normal,constant
import re
import keras_resnet
import keras_resnet.models
from keras.layers import Input, Activation, Conv2D, Conv2DTranspose, Concatenate, Add, UpSampling2D, Lambda, Maximum, \
    Average, Multiply
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from ..layers import UpsampleLike
np = 17
weight_decay = 5e-4
use_mask = True

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    if name is None:
        x = Conv2D(nf, (ks, ks), padding='same',
                   kernel_regularizer=kernel_reg,
                   bias_regularizer=bias_reg,
                   kernel_initializer=random_normal(stddev=0.01),
                   bias_initializer=constant(0.0))(x)
    else:
        x = Conv2D(nf, (ks, ks), padding='same', name=name,
                   kernel_regularizer=kernel_reg,
                   bias_regularizer=bias_reg,
                   kernel_initializer=random_normal(stddev=0.01),
                   bias_initializer=constant(0.0))(x)
    return x

def create_pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = Add(name='P4_merged')([P5_upsampled, P4])
    P4           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    # P7 = Activation('relu', name='C6_relu')(P6)
    # P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x

def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x

def apply_mask(x, mask, stage):
    w_name = "weight_stage%d" % (stage)
    w = Multiply(name=w_name)([x, mask])
    return w

def single_upsample_block(inp, weight_decay):
    out = UpSampling2D()(inp)
    for i in range(3):
        out = conv(out, 256, 3, None, (weight_decay, 0))
    # out = BatchNormalization()(out)
    out = Activation(activation='relu')(out)
    return out

def small_stride_features(features, up, weight_decay):
    up_feats = []
    P3, P4, P5, P6 = features
    nb = 256
    k = 3

    x = conv(P3, nb, k, None, (weight_decay, 0))
    x = relu(x)
    up_feats.append(x)

    x = up(P4, weight_decay)
    x = conv(x, nb, k, None, (weight_decay, 0))
    x = relu(x)
    up_feats.append(x)

    x = up(P5, weight_decay)
    x = up(x, weight_decay)
    x = conv(x, nb, k, None, (weight_decay, 0))
    x = relu(x)
    up_feats.append(x)

    x = up(P6, weight_decay)
    x = up(x, weight_decay)
    x = up(x, weight_decay)
    x = conv(x, nb, k, None, (weight_decay, 0))
    x = relu(x)
    up_feats.append(x)

    x = Concatenate()(up_feats)
    return x

def build_model():
    inputs = []
    outputs = []

    weights_path = '/home/muhammed/.keras/models/ResNet-101-model.keras.h5'

    image_input = Input((None, None, 3))
    heat_weight_input = Input(shape=(None, None, 17))

    inputs.append(image_input)
    inputs.append(heat_weight_input)

    backbone = keras_resnet.models.ResNet101(image_input, include_top=False, freeze_bn=True)

    print 'Loading weights from %s' % weights_path
    backbone.load_weights(weights_path, by_name=True)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2
    features = create_pyramid_features(C3, C4, C5)

    x = small_stride_features(features, single_upsample_block, weight_decay)

    # Additional non rpn layers
    x = conv(x, 512, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_5_CPM", (weight_decay, 0))
    feats = relu(x)

    stages = 6

    # stage 1
    stage1_out = stage1_block(feats, np, 1, weight_decay)

    if use_mask:
        w = apply_mask(stage1_out, heat_weight_input, 1)
    else:
        w_name = "weight_stage%d" % (1)
        w = Activation(activation='linear', name=w_name)(stage1_out)

    x = Concatenate()([stage1_out, feats])
    outputs.append(w)

    for sn in range(2, stages + 1):
        stageT_out = stageT_block(x, np, sn, 1, weight_decay)

        if use_mask:
            w = apply_mask(stageT_out, heat_weight_input, sn)
        else:
            w_name = "weight_stage%d" % (sn)
            w = Activation(activation='linear', name=w_name)(stageT_out)

        outputs.append(w)

        if sn < stages:
            x = Concatenate()([stageT_out, feats])

    if use_mask:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = Model(inputs=image_input, outputs=outputs)

    return model

def get_prediction_model():

    weight_decay = 5e-4

    weights_path = '/home/muhammed/.keras/models/ResNet-101-model.keras.h5'

    image_input = Input((None, None, 3))

    backbone = keras_resnet.models.ResNet101(image_input, include_top=False, freeze_bn=True)

    print 'Loading weights from %s' % weights_path
    backbone.load_weights(weights_path, by_name=True)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2
    features = create_pyramid_features(C3, C4, C5)

    x = small_stride_features(features, single_upsample_block, weight_decay)

    # Additional non rpn layers
    x = conv(x, 512, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_5_CPM", (weight_decay, 0))
    feats = relu(x)

    stages = 6

    # stage 1
    stage1_out = stage1_block(feats, np, 1, weight_decay)

    x = Concatenate()([stage1_out, feats])

    for sn in range(2, stages + 1):
        stageT_out = stageT_block(x, np, sn, 1, weight_decay)

        if sn < stages:
            x = Concatenate()([stageT_out, feats])

    model = Model(inputs=image_input, outputs=stageT_out)

    return model


def setup_lr_mult(model):
    lr_mult = dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):
            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2
            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

    return lr_mult