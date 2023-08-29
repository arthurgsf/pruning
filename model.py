import tensorflow as tf
import segmentation_models as sm

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

    output = unet(conv)

    return tf.keras.Model(input, output, name = "EfficientUnet")


# def EfficientDeeplab(input_shape:tuple, backbone:Backbones):


#     base_model, layers, _ = tasm.create_base_model(
#         name    =   f"efficientnet{backbone.name}".lower(),
#         height  =   input_shape[1], 
#         width   =   input_shape[0])
    
#     deeplab = tasm.DeepLabV3plus(
#         n_classes           =   1,
#         base_model          =   base_model,
#         output_layers       =   layers,
#         backbone_trainable  =   True,
#         final_activation    =   'sigmoid')
        
#     # map N channels data to 3 channelss
#     inputs = tf.keras.Input(shape=(None, None, 1))
#     conv = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(inputs)
#     outputs = deeplab(conv)

#     return tf.keras.Model(inputs, outputs, name=f"EfficientDeeplab{backbone.name}")