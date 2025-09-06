"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dotenv import dotenv_values
from utilities import inference_class

# Declare constants
SEED = 1
MODEL_NAME = 'tevae'

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

# Iterate through model seeds and folds
for fold_idx in range(3):
    for split in ['1day', '1week', '2weeks', '3weeks', '4weeks']:
        data_load_path = os.path.join(data_path, '2_preprocessed', 'fold_' + str(fold_idx))
        model_name = MODEL_NAME + '_' + split + '_' + str(fold_idx) + '_1'  # Fixed model seed due to focus on query strategy
        model_load_path = os.path.join(model_path, model_name)

        inferencer = inference_class.Inferencer(
            model_path=model_load_path,
            window_size=256,
            window_shift=1,
        )

        # Load data
        train_list = inferencer.load_pickle(os.path.join(data_load_path, split, 'train.pkl'))
        val_list = inferencer.load_pickle(os.path.join(data_load_path, split, 'val.pkl'))
        test_list = inferencer.load_pickle(os.path.join(data_load_path, split, 'test.pkl'))

        # Inference
        subset_name = 'train'
        train_detection_score_list, train_output = inferencer.inference_list(
            train_list,
            subset_name=subset_name,
            save_inference_results=True
        )

        subset_name = 'val'
        val_detection_score_list, val_output = inferencer.inference_list(
            val_list,
            subset_name=subset_name,
            save_inference_results=True
        )

        subset_name = 'test'
        test_detection_score_list, test_output = inferencer.inference_list(
            test_list,
            subset_name=subset_name,
            save_inference_results=True
        )
