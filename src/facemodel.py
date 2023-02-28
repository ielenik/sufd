
import tensorflow as tf
from . import config as conf

INPUT_SIZE = None

def display_model_info(model):
    total_weights = 0
    total_pixels = 0
    for w in model.trainable_weights:
        total_weights += tf.size(w).numpy()
    print("Model total weights", total_weights)

    if not INPUT_SIZE:
        return
    for l in model.layers:
        layer_weights = 0
        for w in l.trainable_weights:
            layer_weights += tf.size(w).numpy()

        shape = l.output_shape
        if len(shape) < 4:
            shape = shape[0]
        total_pixels += shape[1]*shape[2]*shape[3]
        print(l.name, shape[1]*shape[2]*shape[3]*4*2*64/2**20,'mb', layer_weights)
    
    print("Model total pixels", total_pixels)
    print("Model total memory for", conf.BATCH_SIZE ,"batch", total_pixels*4*2*conf.BATCH_SIZE/2**20,'mb')

def createFaceModel():
    def dn_block(x, nm, num_layers, do_max_pool = True):
        for _ in range(num_layers):
            x = tf.keras.layers.Conv2D(nm, 3, padding='same', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
        if do_max_pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        return x

    def up_block(x, y, nm):
        x = tf.keras.layers.UpSampling2D(size = (2,2))(x)
        x = tf.concat([x,y], axis = -1)
        x = tf.keras.layers.Conv2D(nm, 1, padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(nm, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def vgg8_squareface():
        input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

        x = dn_block(input, 16, 1)
        x = dn_block(x, 32, 2)
        x = dn_block(x, 64, 4)
        x = dn_block(x, 128, 4)
        x1 = dn_block(x, 256, 4)
        x2 = dn_block(x1, 512, 4)
        x2 = dn_block(x2, 512, 4, False)
        x1 = up_block(x2, x1, 256)
        x = up_block(x1, x, 128)

        return tf.keras.Model(input, x)

    input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    feathures_model = vgg8_squareface()
    display_model_info(feathures_model)
    feathures = feathures_model(input)
    
    feathures = tf.keras.layers.Dropout(0.4)(feathures)
    face_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    face_prob = tf.keras.layers.BatchNormalization()(face_prob)
    face_prob = tf.keras.layers.Dense(1)(face_prob) - 10
    maskspoof_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    maskspoof_prob = tf.keras.layers.BatchNormalization()(maskspoof_prob)
    maskspoof_prob = tf.keras.layers.Dense(2)(maskspoof_prob)
    metric_regr = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    metric_regr = tf.keras.layers.BatchNormalization()(metric_regr)
    metric_regr = tf.keras.layers.Dense(4)(metric_regr)*0.1
    output = tf.concat([face_prob,metric_regr,maskspoof_prob], axis = -1)
    model = tf.keras.Model(input,output)
    display_model_info(model)
    return model

def createFaceModel4():
    def dn_block(x, nm, num_layers, do_max_pool = True):
        for _ in range(num_layers):
            x = tf.keras.layers.Conv2D(nm, 3, padding='same', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
        if do_max_pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        return x

    def vgg8_squareface():
        input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

        x = dn_block(input, 16, 1)
        x = dn_block(x, 32, 2)
        x = dn_block(x, 64, 4)
        x = dn_block(x, 128, 4)
        return tf.keras.Model(input, x)

    input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    feathures_model = vgg8_squareface()
    display_model_info(feathures_model)
    feathures = feathures_model(input)
    
    feathures = tf.keras.layers.Dropout(0.4)(feathures)
    face_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    face_prob = tf.keras.layers.BatchNormalization()(face_prob)
    face_prob = tf.keras.layers.Dense(1)(face_prob) - 10
    maskspoof_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    maskspoof_prob = tf.keras.layers.BatchNormalization()(maskspoof_prob)
    maskspoof_prob = tf.keras.layers.Dense(2)(maskspoof_prob)
    metric_regr = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    metric_regr = tf.keras.layers.BatchNormalization()(metric_regr)
    metric_regr = tf.keras.layers.Dense(4)(metric_regr)*0.1
    output = tf.concat([face_prob,metric_regr,maskspoof_prob], axis = -1)
    model = tf.keras.Model(input,output)
    display_model_info(model)
    return model

def createFaceModel3():
    def dn_block(x, nm, num_layers, do_max_pool = True):
        for _ in range(num_layers):
            x = tf.keras.layers.Conv2D(nm, 3, padding='same', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
        if do_max_pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        return x

    def up_block(x, y, nm):
        x = tf.keras.layers.UpSampling2D(size = (2,2))(x)
        x = tf.concat([x,y], axis = -1)
        x = tf.keras.layers.Conv2D(nm, 1, padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(nm, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def vgg8_squareface():
        input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

        x = tf.concat([input,tf.square(input)], axis = -1)
        x1 = tf.keras.layers.Conv2D(8, 5, padding='same')(x)
        x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.concat([x,tf.square(x)], axis = -1)
        x1 = tf.keras.layers.Conv2D(16, 3, padding='same')(x)
        x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.concat([x,tf.square(x)], axis = -1)
        x1 = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
        x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.concat([x,tf.square(x)], axis = -1)
        x1 = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

        # x = tf.keras.layers.Conv2D(16, 5, strides=(2, 2), padding='same', activation = 'relu')(input)
        # x = tf.keras.layers.Conv2D(16, 3, strides=(1, 1), padding='same', activation = 'relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same', activation = 'relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = dn_block(x, 64, 3)
        # x = dn_block(x, 128, 4)
        x1 = dn_block(x, 128, 4)
        x2 = dn_block(x1, 256, 4)
        x2 = dn_block(x2, 256, 4, False)
        x1 = up_block(x2,x1, 128)
        x = up_block(x1,x, 128)

        return tf.keras.Model(input, x)

    input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    feathures_model = vgg8_squareface()
    display_model_info(feathures_model)
    feathures = feathures_model(input)
    
    feathures = tf.keras.layers.Dropout(0.4)(feathures)
    face_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    face_prob = tf.keras.layers.BatchNormalization()(face_prob)
    face_prob = tf.keras.layers.Dense(1)(face_prob) - 10
    maskspoof_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    maskspoof_prob = tf.keras.layers.BatchNormalization()(maskspoof_prob)
    maskspoof_prob = tf.keras.layers.Dense(2)(maskspoof_prob)
    metric_regr = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    metric_regr = tf.keras.layers.BatchNormalization()(metric_regr)
    metric_regr = tf.keras.layers.Dense(4)(metric_regr)*0.1
    output = tf.concat([face_prob,metric_regr,maskspoof_prob], axis = -1)
    model = tf.keras.Model(input,output)
    display_model_info(model)
    return model


def createFaceModel2():
    modelUnet = sm.Unet('resnet34', classes=64, activation=None, input_shape=(None, None, 67), encoder_weights=None)
    input = tf.keras.layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = tf.concat([input,tf.square(input)], axis = -1)
    x1 = tf.keras.layers.Conv2D(8, 5, padding='same')(x)
    x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.concat([x,tf.square(x)], axis = -1)
    x1 = tf.keras.layers.Conv2D(16, 3, padding='same')(x)
    x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.concat([x,tf.square(x)], axis = -1)
    x1 = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.concat([x,tf.square(x)], axis = -1)
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.concat([x1,x[:,:,:,:3]], axis = -1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    feathures = modelUnet(x)
    display_model_info(modelUnet)
    feathures = tf.keras.layers.Dropout(0.4)(feathures)
    face_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    face_prob = tf.keras.layers.BatchNormalization()(face_prob)
    face_prob = tf.keras.layers.Dense(1)(face_prob)
    maskspoof_prob = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    maskspoof_prob = tf.keras.layers.BatchNormalization()(maskspoof_prob)
    maskspoof_prob = tf.keras.layers.Dense(2)(maskspoof_prob)
    metric_regr = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(feathures)
    metric_regr = tf.keras.layers.BatchNormalization()(metric_regr)
    metric_regr = tf.keras.layers.Dense(4)(metric_regr)
    output = tf.concat([face_prob,metric_regr,maskspoof_prob], axis = -1)/10.
    model = tf.keras.Model(input,output)
    display_model_info(model)
    return model


    # input = tf.keras.layers.Input(shape=(None, None, 3))
    # x = tf.concat([input,tf.square(input)], axis = -1)
    # x = tf.keras.layers.Conv2D(16, 5, padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(64, 3, padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
