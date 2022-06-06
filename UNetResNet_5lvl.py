from keras.models import Model
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from utils.blocks import conv2d_block, residual_block


def get_unet(input_img, n_filters=8, kernel_size = 3, activation = 'relu'):
    """Funkcja definiująca model o architekturze U-Net z blokami resztkowymi."""
    #Ścieżka kurcząca
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size, activation)
    r1 = residual_block(c1, n_filters * 1,activation, kernel_size)
    cc1 = conv2d_block(r1, n_filters * 1, kernel_size, activation)
    p1 = MaxPooling2D((2, 2), padding='same')(cc1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size, activation)
    r2 = residual_block(c2, n_filters * 2,activation, kernel_size)
    cc2 = conv2d_block(r2, n_filters * 2, kernel_size, activation)
    p2 = MaxPooling2D((2, 2), padding='same')(cc2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size, activation)
    r3 = residual_block(c3, n_filters * 4,activation, kernel_size)
    cc3 = conv2d_block(r3, n_filters * 4, kernel_size, activation)
    p3 = MaxPooling2D((2, 2), padding='same')(cc3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size, activation)
    r4 = residual_block(c4, n_filters * 8,activation, kernel_size)
    cc4 = conv2d_block(r4, n_filters * 8, kernel_size, activation)
    p4 = MaxPooling2D((2, 2), padding='same')(cc4)

    c5 = conv2d_block(p4, n_filters * 16, kernel_size, activation)
    r5 = residual_block(c5, n_filters * 16,activation, kernel_size)
    cc5 = conv2d_block(r5, n_filters * 16, kernel_size, activation)
    p5 = MaxPooling2D((2, 2), padding='same')(cc5)

    c6 = conv2d_block(p5, n_filters*32, kernel_size, activation)
    r6 = residual_block(c6, n_filters * 32,activation, kernel_size)
    cc6 = conv2d_block(r6, n_filters * 32, kernel_size, activation)
    # Ścieżka ekspansywna
    u6 = Conv2DTranspose(n_filters * 16, (3, 3), strides=(2, 2), padding='same')(cc6)
    u6 = concatenate([u6, cc5])
    c7 = conv2d_block(u6, n_filters * 16, kernel_size, activation)
    r7 = residual_block(c7, n_filters * 16,activation, kernel_size)
    cc7 = conv2d_block(r7, n_filters * 16, kernel_size, activation)

    u7 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(cc7)
    u7 = concatenate([u7, cc4])
    c8 = conv2d_block(u7, n_filters * 8, kernel_size, activation)
    r8 = residual_block(c8, n_filters * 8,activation, kernel_size)
    cc8 = conv2d_block(r8, n_filters * 8, kernel_size, activation)

    u8 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(cc8)
    u8 = concatenate([u8, cc3])
    c9 = conv2d_block(u8, n_filters * 4, kernel_size, activation)
    r9 = residual_block(c9, n_filters * 4,activation, kernel_size)
    cc9 = conv2d_block(r9, n_filters * 4, kernel_size, activation)

    u9 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(cc9)
    u9 = concatenate([u9, cc2])
    c10= conv2d_block(u9, n_filters * 2, kernel_size, activation)
    r10 = residual_block(c10, n_filters * 2,activation, kernel_size)
    cc10 = conv2d_block(r10, n_filters * 2, kernel_size, activation)

    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(cc10)
    u10 = concatenate([u10, cc1])
    c11 = conv2d_block(u10, n_filters * 1, kernel_size, activation)
    r11 = residual_block(c11, n_filters * 1,activation, kernel_size)
    cc11 = conv2d_block(r11, n_filters * 1, kernel_size, activation)

    outputs = Conv2D(1, (1, 1), activation='relu')(cc11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model