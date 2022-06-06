from keras.layers import  BatchNormalization, Activation, add,PReLU, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers import concatenate




def conv2d_block(input_tensor, n_filters, kernel_size, activation, kernel_regularizer = None):
    """Funkcja definiująca pojedynczą operację konwolucji. Została ona utworzona w celu ułatwienia
    czytelności kodu."""

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same',kernel_regularizer=kernel_regularizer)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)



    return x

def residual_block(input_tensor, n_filters, activation='relu',kernel_size = 3,  kernel_regularizer = None):
    """Funkcja definiująca blok resztkowy jako dwa bloki konwolucyjne i połącznenie skrótowe."""

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same',kernel_regularizer=kernel_regularizer)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)


    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same',kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = add([input_tensor, x])
    return x


def initial_conv(input_tensor, n_filters, padding = 'same'):
    c1 = Conv2D(filters=n_filters, kernel_size=(1,1), padding= padding)(input_tensor)
    return c1


def conv2D_3x3_s2(input_tensor, n_filters, padding = 'same', input_shape = None):
    c1 = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(2,2), padding = padding)(input_tensor)

    bn = BatchNormalization()(c1)
    pr = PReLU()(bn)
    return pr

def conv2D_3x3_s1(input_tensor, n_filters, padding = 'same'):
    c1 = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding = padding)(input_tensor)

    bn = BatchNormalization()(c1)
    pr = PReLU()(bn)
    return pr

def conv2D_1x1_s1(input_tensor, n_filters):
    c1 = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1))(input_tensor)

    bn = BatchNormalization()(c1)
    pr = PReLU()(bn)
    return pr

def bilinear_interp_Upsampling_2x2(input_tensor):
    u1 = UpSampling2D(interpolation='bilinear')(input_tensor)

    return u1

def skip_connection_1x1(encoder_tensor, decoder_tensor):
    c1 = Conv2D(filters = 4, kernel_size=(1,1))(encoder_tensor)
    con1 = concatenate([c1, decoder_tensor])

    return con1