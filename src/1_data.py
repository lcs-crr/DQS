"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dotenv import dotenv_values
from utilities import data_class

# Declare constants
SEED = 1
FREQ_TARGET = 2
FREQ_ORIG = 10

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

data_processor = data_class.DataProcessor(
    original_sampling_rate=FREQ_ORIG,
    target_sampling_rate=FREQ_TARGET,
    scale_method='z-score'
)

# Specify paths for loading and saving data
data_load_path = os.path.join(data_path, '1_postsim')
data_save_path = os.path.join(data_path, '2_preprocessed')
os.makedirs(data_save_path, exist_ok=True)

train_list_fold = data_processor.load_pickle(os.path.join(data_load_path, 'train.pkl'))
test_list_fold = data_processor.load_pickle(os.path.join(data_load_path, 'test.pkl'))

for fold_idx, _ in enumerate(train_list_fold):
    # Downsample data lists
    train_list_resampled = data_processor.downsample_list(train_list_fold[fold_idx])
    test_list_resampled = data_processor.downsample_list(test_list_fold[fold_idx])

    split_idcs = data_processor.split_into_hours(train_list_resampled, [1, 7, 14, 21, 28])
    split_names = ['1day', '1week', '2weeks', '3weeks', '4weeks']

    os.makedirs(os.path.join(data_save_path, 'fold_' + str(fold_idx)), exist_ok=True)
    file = open(os.path.join(data_save_path, 'fold_' + str(fold_idx), 'split_idcs.txt'), 'w')
    for split_idx, split in enumerate(split_idcs):
        file.write(str(split) + "\n")
    file.close()

    for split_idx, split in enumerate(split_idcs):
        # Split training data into training and validation
        train_list_resampled_split, val_list_resampled_split = train_test_split(train_list_resampled[:split], random_state=SEED, test_size=0.2)

        # Find the scalers for each feature
        data_processor.find_scalers_from_list(train_list_resampled_split)

        # Scale data
        train_list_scaled = data_processor.scale_list(train_list_resampled_split)
        val_list_scaled = data_processor.scale_list(val_list_resampled_split)
        test_list_scaled = data_processor.scale_list(test_list_resampled)

        # Find the window size
        data_processor.find_window_size_from_list(train_list_scaled)

        # Window sequences inside lists
        scaled_train_window = data_processor.window_list(train_list_scaled)
        scaled_val_window = data_processor.window_list(val_list_scaled)

        # Create tf.data objects
        tfdata_train = tf.data.Dataset.from_tensor_slices(scaled_train_window)
        tfdata_val = tf.data.Dataset.from_tensor_slices(scaled_val_window)

        # Shuffle and batch tf.data objects
        tfdata_train = tfdata_train.shuffle(tfdata_train.cardinality(), seed=SEED)
        tfdata_val = tfdata_val.shuffle(tfdata_val.cardinality(), seed=SEED)

        # Save training and validation data as tf.data
        tf.data.Dataset.save(tfdata_train, os.path.join(data_save_path, 'fold_' + str(fold_idx), split_names[split_idx], 'train'))
        tf.data.Dataset.save(tfdata_val, os.path.join(data_save_path, 'fold_' + str(fold_idx), split_names[split_idx], 'val'))

        # Save training, validation and testing data as pickle files
        data_processor.dump_pickle(train_list_scaled, os.path.join(data_save_path, 'fold_' + str(fold_idx), split_names[split_idx], 'train.pkl'))
        data_processor.dump_pickle(val_list_scaled, os.path.join(data_save_path, 'fold_' + str(fold_idx), split_names[split_idx], 'val.pkl'))
        data_processor.dump_pickle(test_list_scaled, os.path.join(data_save_path, 'fold_' + str(fold_idx), split_names[split_idx], 'test.pkl'))
