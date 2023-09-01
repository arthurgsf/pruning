import os
import pickle
import preprocessing
from glob import glob
import tensorflow as tf
from model import EfficientUnet
import segmentation_models as sm
from generators import RandomPatientSliceGenerator

DATASET_PATH = f"{os.path.expanduser('~')}/Datasets/segthor_extracted"
IMG_SHAPE = (256, 256)
N_CHANNELS = 1
INPUT_SHAPE = IMG_SHAPE + (N_CHANNELS,)
N_CLASSES = 1
OUTPUT_SHAPE = IMG_SHAPE + (N_CLASSES,)
BATCH_SIZE = 20
EPOCHS = 100

if __name__ == "__main__":
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    # train - val
    tmp_train_patients = glob(f"{DATASET_PATH}/train/Patient*/")
    idx_train = int(len(tmp_train_patients)*0.9)
    train_patients = tmp_train_patients[:idx_train] # ~90% da base (25 pacientes)
    val_patients = tmp_train_patients[idx_train + 1:] # ~10% da base (3 pacientes)

    preprocessing_pipeline = preprocessing.Pipeline([
        preprocessing.windowing(-500, 60),
        preprocessing.norm, 
        preprocessing.resize(IMG_SHAPE),
        preprocessing.expand_dims
    ])

    # generators
    train_dataset = tf.data.Dataset.from_generator(
        RandomPatientSliceGenerator(train_patients, preprocessing_pipeline),
        output_signature =
        (
            tf.TensorSpec(shape=INPUT_SHAPE),
            tf.TensorSpec(shape=OUTPUT_SHAPE)
        ),
    ).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_generator(
        RandomPatientSliceGenerator(val_patients, preprocessing_pipeline),
        output_signature=
        (
            tf.TensorSpec(shape=INPUT_SHAPE),
            tf.TensorSpec(shape=OUTPUT_SHAPE)
        )
    ).batch(BATCH_SIZE)

    # model setup
    model = EfficientUnet(input_shape = INPUT_SHAPE)

    # metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    precision_score = tf.keras.metrics.Precision(name="precision", thresholds=0.5)
    recall_score = tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    iou_score = sm.metrics.IOUScore(threshold=0.5)

    # callbacks
    checkpoint_dir = f"records/{model.name}"
    os.makedirs(checkpoint_dir+"/checkpoint", exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_dir}/checkpoint/{model.name}",
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True,
        save_freq='epoch',
        mode='min'
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode='min',
        factor=0.3,
        patience=5,
        cooldown=5,
        min_lr = 0.00001,
        min_delta=0.01
        )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode='min',
        restore_best_weights=True,
        patience=10,
        min_delta=0.01
    )

    # compile
    model.compile(optimizer, 
            loss      =   sm.losses.DiceLoss(),
            metrics   =   [
                precision_score,
                recall_score,
            ],
            weighted_metrics = [
                iou_score,
                f_score,      
            ])
    
    # model setup
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
    
    # fit
    H = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, lr_scheduler])

    with open(MODEL_PATH + "/history", "wb") as f:
        pickle.dump(H.history, f)