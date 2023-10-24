import tensorflow as tf
import segmentation_models as sm

def EfficientUnet(input_shape:tuple, n_classes = 1):
    input = tf.keras.layers.Input(shape=input_shape)
    conv = tf.keras.layers.Conv2D(filters = 3, kernel_size = 1, activation='relu')(input)
    
    unet = sm.Unet(
        'efficientnetb0',
        input_shape=(input_shape[0], input_shape[1], 3),
        classes=n_classes, 
        #encoder_weights='imagenet',
        activation='sigmoid'
    )

    output = unet(conv)

    return tf.keras.Model(input, output, name = "EfficientUnet")