import random
from typing import Any
import cv2 as cv
import numpy as np
from glob import glob
from pathlib import Path
import tensorflow as tf
from preprocessing import Pipeline
import pandas as pd

class PerPatientSliceGenerator:
    def __init__(self, patient, preprocessing_pipeline:Pipeline):
        self.pipeline = preprocessing_pipeline
        self.patient = patient
    
    def __call__(self) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        generate ordered slices for a single patient
        """
        annotations = [f for f in glob(f"{self.patient}/*.png") if "GT_" in f ]
        annotations.sort(key = lambda p: int(Path(p).name.replace("GT_", "").replace(".png", "")))
        img = [f.replace("GT_", "").replace(".png", ".pfm") for f in annotations]
        
        for i in range(len(annotations)):
            x = cv.imread(img[i], cv.IMREAD_UNCHANGED)
            y = cv.imread(annotations[i], cv.IMREAD_UNCHANGED)
            y = np.clip(y, 0 , 1)

            if self.pipeline:
                x, y = self.pipeline.apply(x, y)

            yield x.astype(np.float32), y.astype(np.float32)

class RandomPatientSliceGenerator:
    def __init__(self, patients, preprocessing_pipeline:Pipeline, balanced=True):
        self.pipeline = preprocessing_pipeline
        self.patients = patients
        self.balanced = balanced
    
    def __call__(self) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        info_dataframes = []
        for patient_folder in self.patients:
            info = pd.read_csv(f"{patient_folder}/info.csv")
            with_trachea = info[info["contains_trachea"] == True].reset_index()
            info_dataframes.append(with_trachea)
            with_trachea["image_file"] = with_trachea["image_file"].apply(lambda filename:f"{patient_folder}/{filename}")
            # without_trachea = info[info["contains_trachea"] == False]
            # without_trachea = without_trachea.sample(with_trachea.shape[0])

            # balanced_info = pd.concat([with_trachea, without_trachea], axis = 0)
            # balanced_info = balanced_info.sample(frac=1).reset_index()
        info_dataframes = pd.concat(info_dataframes, axis = 0).sample(frac=1).reset_index()

        for i in range(info_dataframes.shape[0]):
            
            p = Path(info_dataframes["image_file"][i])
            x = cv.imread(p.with_suffix(".pfm").as_posix(), cv.IMREAD_UNCHANGED)
            y = cv.imread(p.with_name(f"GT_{p.with_suffix('.png').name}").as_posix(), cv.IMREAD_UNCHANGED)
            y = np.clip(y, 0 , 1)
            
            if self.pipeline:
                x, y = self.pipeline.apply(x, y)
                
            yield x.astype(np.float32), y.astype(np.float32)

# def get_train_val(
#         dataset_path, 
#         image_shape,
#         n_classes, 
#         batch_size=None, 
#         dtype=tf.float32):
    
#     tmp_train_patients = glob(f"{dataset_path}/train/Patient*/")
#     idx_train = int(len(tmp_train_patients)*0.9)
#     train_patients = tmp_train_patients[:idx_train] # ~90% da base (25 pacientes)
#     val_patients = tmp_train_patients[idx_train + 1:] # ~10% da base (3 pacientes)

#     train_dataset = tf.data.Dataset.from_generator(
#         random_slices,
#         output_signature=
#         (
#             tf.TensorSpec(shape=image_shape, dtype=dtype),
#             tf.TensorSpec(shape=image_shape +(n_classes,), dtype=dtype),
#             tf.TensorSpec(shape=(2,), dtype = dtype)
#         ),
#         args = (train_patients, )
#     )

#     val_dataset = tf.data.Dataset.from_generator(
#         random_slices,
#         output_signature=
#         (
#             tf.TensorSpec(shape=image_shape, dtype=dtype),
#             tf.TensorSpec(shape=image_shape +(n_classes,), dtype=dtype),
#             tf.TensorSpec(shape=(2,), dtype = dtype)
#         ),
#         args = (val_patients, )
#     )
    
#     if batch_size:
#         train_dataset = train_dataset.batch(batch_size)
#         val_dataset = val_dataset.batch(batch_size)
#     return train_dataset, val_dataset