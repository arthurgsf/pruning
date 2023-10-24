import os
import pickle
import numpy as np
from glob import glob
import optuna as tuna
import tensorflow as tf
from model import EfficientUnet
import segmentation_models as sm
from segthor_generators import RandomPatientSliceGenerator, PatientSliceGenerator
from config import args
import tfpreprocessing as tfp
import preprocessing as pre
import utils


def train(trial, preprocessing_pipeline):
    # split : train/val
    tmp_train_patients = glob(f"{args.train_dir}/Patient*/")
    idx_train = int(len(tmp_train_patients)*0.9)
    train_patients = tmp_train_patients[:idx_train] # ~90% da base (25 pacientes)
    val_patients = tmp_train_patients[idx_train:] # ~10% da base (3 pacientes)

    # generators setup
    train_dataset = tf.data.Dataset.from_generator(
        RandomPatientSliceGenerator(train_patients, preprocessing_pipeline),
        output_signature =
        (
            tf.TensorSpec(shape=args.image_shape + (args.n_channels,)),
            tf.TensorSpec(shape=args.image_shape + (args.n_classes,)),
            tf.TensorSpec(shape=(2,))
        ),
    ).batch(args.batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        RandomPatientSliceGenerator(val_patients, preprocessing_pipeline),
        output_signature=
        (
            tf.TensorSpec(shape=args.image_shape + (args.n_channels,)),
            tf.TensorSpec(shape=args.image_shape + (args.n_classes,)),
            tf.TensorSpec(shape=(2,))
        )
    ).batch(args.batch_size)

    # model setup
    model = EfficientUnet(
        input_shape = args.input_shape + (args.n_channels,),
        n_classes = args.n_classes
    )

    # metrics setup
    lr = trial.suggest_float("learning_rate", 0.0001, 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    precision_score = tf.keras.metrics.Precision(name="precision", thresholds=0.5)
    recall_score = tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    iou_score = sm.metrics.IOUScore(threshold=0.5)

    # callbacks setup
    MODEL_PATH = utils.model_path(model, trial)
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
        factor      =   0.1,
        patience    =   5,
        cooldown    =   5,
        min_lr      =   0.000001,
        min_delta   =   0.001
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=10
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

    H = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[checkpoint, lr_scheduler, early_stopping]
        )
    
    utils.save_history(H.history, MODEL_PATH)

    return model

def test(model, trial, preprocessing_pipeline):
    MODEL_PATH = utils.get_model_path(model, trial)
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

    patients_test = glob(args.test_dir + '/*')

    for patient in patients_test:
        patient_dataset = tf.data.Dataset.from_generator(
            PatientSliceGenerator(patient, preprocessing_pipeline),
            output_signature=
            (
                tf.TensorSpec(shape=args.input_shape + (args.n_channels,), dtype=tf.float32),
                tf.TensorSpec(shape=args.input_shape + (args.n_classes), dtype=tf.float32)
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

    utils.store_test_metrics(metrics, MODEL_PATH)
    
    return np.mean(metrics["dice"])

def obj(trial:tuna.Trial):

    # hyper-parameters
    lower_bound = trial.suggest_int("lower_bound", -100, 0)
    upper_bound = trial.suggest_int("upper_bound", 1, 80)

    preprocessing_pipeline = tfp.Pipeline([
        pre.windowing(lower_bound, upper_bound),
        pre.resize(args.input_shape),
        pre.norm(),
        pre.expand_dims()
    ])

    model = train(trial, preprocessing_pipeline)
    dice_score = test(model, trial, preprocessing_pipeline)

    with open(f"{utils.get_model_path(model, trial)}/params", "rb") as f:
        pickle.dump(trial.params, f)

    return dice_score

if __name__ == "__main__":
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    study = tuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=args.opt_epochs)