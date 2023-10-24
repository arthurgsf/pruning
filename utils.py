import pickle
import optuna as tuna
import tensorflow as tf
from config import args

def get_model_path(model:tf.keras.Model, trial:tuna.Trial):
    return f"{args.records_dir}/{model.name}/{trial.number}/"

def save_history(history, model_path):
    with open(f"{model_path}/history", "wb") as f:
            pickle.dump(history, f)

def store_test_metrics(metrics, model_path):    
    with open(f"{model_path}/test_metrics", "wb") as f:
        pickle.dump(metrics, f)