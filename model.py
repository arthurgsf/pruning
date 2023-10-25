import tensorflow as tf
import segmentation_models as sm

def build_regularizer(kernel, regularizer):
    return lambda: regularizer(kernel)


def EfficientUnet(input_shape:tuple, n_classes = 1):
    input = tf.keras.layers.Input(shape=input_shape)
    conv = tf.keras.layers.Conv2D(filters = 3, kernel_size = 1)(input)
    
    unet = sm.Unet(
        'efficientnetb0',
        input_shape=(input_shape[0], input_shape[1], 3),
        classes=n_classes,
        encoder_weights='imagenet',
        activation='sigmoid'
    )

    # add weight regularizer to stabilize training curve
    alpha = 0.00002  # weight decay coefficient
    regularizer = tf.keras.regularizers.l2(alpha)

    for layer in unet.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(build_regularizer(layer.kernel, regularizer))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(build_regularizer(layer.bias, regularizer))

    output = unet(conv)

    return tf.keras.Model(input, output, name = "EfficientUnet")