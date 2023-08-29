import os
import pickle
import tensorflow as tf

def build_model(input_shape = (224, 224, 3), n_classes=10):
    vgg = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape)
    flatten = tf.keras.layers.GlobalAveragePooling2D()(vgg.output)
    dense_1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
    dense_2 = tf.keras.layers.Dense(256, activation="relu")(dense_1)
    output = tf.keras.layers.Dense(n_classes, activation="softmax")(dense_2)
    return tf.keras.Model(vgg.input, output, name=f"OxfordFlowers102-{vgg.name}")

if __name__ == "__main__":
    import tensorflow_datasets as tfds
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    # dataset setup
    (train_dataset, test_dataset, validation_dataset), ds_info = tfds.load(
        "oxford_flowers102",
        split=["train", "test", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def preprocess(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        return tf.cast(image, tf.float32) / 255.0, label

    train_dataset = train_dataset.map(preprocess)
    
    train_dataset = train_dataset.shuffle(ds_info.splits["train"].num_examples)
    train_dataset = train_dataset.batch(32)
    
    validation_dataset = validation_dataset.map(preprocess)
    validation_dataset = validation_dataset.batch(32)

    test_dataset = test_dataset.map(preprocess)
    test_dataset = test_dataset.batch(32)
    
    # model setup

    model = build_model(n_classes=102)

    MODEL_PATH = f"records/{model.name}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/checkpoint"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{CHECKPOINT_PATH}/{model.name}",
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True,
        save_freq='epoch',
        mode='min'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    H = model.fit(
        train_dataset,
        epochs=3,
        validation_data=validation_dataset,
        callbacks=[checkpoint]
    )

    with open(MODEL_PATH + "/history", "wb") as f:
        pickle.dump(H.history, f)