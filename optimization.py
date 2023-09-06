import os
import pickle
import numpy as np
import preprocessing
from glob import glob
import optuna as tuna
import tensorflow as tf
from model import EfficientUnet
import segmentation_models as sm
from generators import RandomPatientSliceGenerator, PerPatientSliceGenerator

DATASET_PATH = f"{os.path.expanduser('~')}/Datasets/segthor_extracted"
IMG_SHAPE = (256, 256)
N_CHANNELS = 1
INPUT_SHAPE = IMG_SHAPE + (N_CHANNELS,)
N_CLASSES = 1
OUTPUT_SHAPE = IMG_SHAPE + (N_CLASSES,)
BATCH_SIZE = 20
EPOCHS = 100

def model_path(model:tf.keras.Model, trial:tuna.Trial):
    return f"opt_records/{model.name}/{trial.number}/"

def train(trial:tuna.Trial, preprocessing_pipeline):
    # split : train/val
    tmp_train_patients = glob(f"{DATASET_PATH}/train/Patient*/")
    idx_train = int(len(tmp_train_patients)*0.9)
    train_patients = tmp_train_patients[:idx_train] # ~90% da base (25 pacientes)
    val_patients = tmp_train_patients[idx_train + 1:] # ~10% da base (3 pacientes)

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
    lr = trial.suggest_float("learning_rate", 0.0001, 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    precision_score = tf.keras.metrics.Precision(name="precision", thresholds=0.5)
    recall_score = tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    iou_score = sm.metrics.IOUScore(threshold=0.5)

    # callbacks
    MODEL_PATH = model_path(model, trial)
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

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode='min',
        factor=0.3,
        patience=5,
        cooldown=5,
        min_lr = 0.00001,
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

    # fit
    H = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, lr_scheduler],
        verbose=0)
    
    with open(MODEL_PATH + "/history", "wb") as f:
        pickle.dump(H.history, f)

    return model


def test(model, trial, preprocessing_pipeline):
    MODEL_PATH = model_path(model, trial)
    model.load_weights(f"{MODEL_PATH}/checkpoint/{model.name}")

    dice_score = sm.metrics.FScore(threshold=0.5)
    precision_score = tf.keras.metrics.Precision(thresholds=0.5)
    recall_score = tf.keras.metrics.Recall(thresholds=0.5)
    iou_score = sm.metrics.IOUScore(threshold=0.5)

    metrics = {
        "dice":[],
        "precision":[],
        "recall":[],
        "iou_score":[],
    }

    patients_test = glob(DATASET_PATH + '/test/*')

    for patient in patients_test:
        patient_dataset = tf.data.Dataset.from_generator(
            PerPatientSliceGenerator(patient, preprocessing_pipeline),
            output_signature=
            (
                tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
                tf.TensorSpec(shape=OUTPUT_SHAPE, dtype=tf.float32)
            )
        )

        volume_true = []
        for (x, y) in patient_dataset:
            volume_true.append(y)
        volume_true = np.squeeze(np.array(volume_true))
        volume_pred = model.predict(patient_dataset.batch(20), verbose=0)
        volume_pred = np.round(volume_pred)
        volume_pred = np.squeeze(volume_pred)

        metrics["dice"].append(dice_score(volume_true, volume_pred))
        metrics["precision"].append(precision_score(volume_true, volume_pred))
        metrics["recall"].append(recall_score(volume_true, volume_pred))
        metrics["iou_score"].append(iou_score(volume_true, volume_pred))

    with open(f"{MODEL_PATH}/test_metrics", "wb") as f:
        pickle.dump(metrics, f)
    
    return np.mean(metrics["dice"])

def obj(trial:tuna.Trial):

    # hyper-parameters
    lower_bound = trial.suggest_int("lower_bound", -500, 0)
    upper_bound = trial.suggest_int("upper_bound", 1, 80)

    preprocessing_pipeline = preprocessing.Pipeline([
        preprocessing.windowing(lower_bound, upper_bound),
        preprocessing.norm, 
        preprocessing.resize(IMG_SHAPE),
        preprocessing.expand_dims
    ])

    model = train(trial, preprocessing_pipeline)
    dice_score = test(model, trial, preprocessing_pipeline)
    return dice_score

if __name__ == "__main__":
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    study = tuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=100)