import numpy as np
import tensorflow as tf
from train import build_model
import tensorflow_datasets as tfds
from kerassurgeon.operations import delete_channels

def lw_biggest_layer(model):
    """
    Returns the biggest prunable layer. If no prunable layer is found, return None.
    """
    prunable = [
        x
        for x in model.layers
        if ("conv" in x.name or "dense" in x.name) and model.output is not x.output
    ]
    prunable = [x for x in prunable if x.weights[0].shape[-1] > 1]

    if len(prunable) > 0:
        total_params = np.array(
            [
                np.sum(np.size(layer.weights[0])) + np.sum(np.size(layer.weights[1]))
                for layer in prunable
            ]
        )
        ix = np.argsort(total_params)[-1]
        return prunable[ix]
    else:
        return None

def lw_prune_layer(model, layer, criteria="l1"):
    """
    Delete the least contributing channel of the passed layer
    """
    idx_to_prune = []

    kernel_weights = layer.weights[0]

    if criteria == "l1":
        if type(layer) is tf.keras.layers.Dense:
            l1_norms = dense_l1_norms(kernel_weights)
        else:
            l1_norms = conv_l1_norms(kernel_weights)
        idx_to_prune = l1_norms.argsort()[:1]

    pruned = delete_channels(model, layer, idx_to_prune, copy=True)
    return pruned


def conv_l1_norms(kernel_weights) -> np.ndarray:
    """
    :param kernel_weights:
    :return:
    """
    num_channels = kernel_weights.shape[-1]
    l1_norms = np.zeros(num_channels)
    for i in range(num_channels):
        l1_norms[i] = np.sum(abs(kernel_weights[:, :, :, i]))

    return l1_norms


def dense_l1_norms(kernel_weights) -> np.ndarray:
    """
    Gets the l1 norms for the given layer (Dense)
    :param kernel_weights:
    :return:
    """
    num_channels = kernel_weights.shape[1]
    # num_weights = kernel_weights.shape[0] - inutil (por enquanto)
    l1_norms = np.zeros(num_channels)

    for i in range(num_channels):
        l1_norms[i] = np.sum(abs(kernel_weights[:, i]))
    return l1_norms

if __name__ == "__main__":
    # TFDS fornecer imagens do tipo tf.uint8 , enquanto o modelo de espera tf.float32 . Portanto, você precisa normalizar as imagens.
    (ds_train, ds_test, ds_valid), ds_info = tfds.load(
        "oxford_flowers102",
        split=["train", "test", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(32)

    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(32)
    
    ds_valid = ds_valid.map(normalize_img)
    ds_valid = ds_valid.batch(32)
    
    model = build_model(input_shape=(224,224,3), n_classes=102)
    model.load_weights(f'records/{model.name}/checkpoint/{model.name}')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    layer = lw_biggest_layer(model)
    while layer:
      pruned = lw_prune_layer(model, layer, criteria="l1")

      # recompilation needed
      pruned.compile(
          optimizer=tf.keras.optimizers.Adam(),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=["accuracy"],
      )

      pruned.fit(
          ds_train,
          epochs=3,
          validation_data=ds_valid,
      )

      # model = pruned
      layer = lw_biggest_layer(pruned)
