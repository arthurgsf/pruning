import os
from glob import glob
import tensorflow as tf
import tfpreprocessing as tfp
from model import EfficientUnet
import segmentation_models as sm
from segthor_generators import RandomPatientSliceGenerator
from config import args
import preprocessing as pre
import utils

def train(preprocessing_pipeline):
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
            tf.TensorSpec(shape=args.image_shape + (args.n_classes,))
            
        ),
    ).batch(args.batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        RandomPatientSliceGenerator(val_patients, preprocessing_pipeline),
        output_signature=
        (
            tf.TensorSpec(shape=args.image_shape + (args.n_channels,)),
            tf.TensorSpec(shape=args.image_shape + (args.n_classes,))
            
        )
    ).batch(args.batch_size)

    # model setup
    model = EfficientUnet(
        input_shape = args.input_shape + (args.n_channels,),
        n_classes = args.n_classes
    )

    # metrics setup
    lr = 0.0008
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    precision_score = tf.keras.metrics.Precision(name="precision", thresholds=0.5)
    recall_score = tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    iou_score = sm.metrics.IOUScore(threshold=0.5)

    # callbacks setup
    MODEL_PATH = f"{args.records_dir}/{model.name}/best/"
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
        patience=5
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

if __name__ == "__main__":
    # tf setup
    devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)                                                                                                                                                                                                                                                                                                          

    preprocessing_pipeline = tfp.Pipeline([
        pre.resize(args.input_shape),
        pre.windowing(-89, 66),
        pre.norm(),
        pre.expand_dims(),
    ])

    train(preprocessing_pipeline)