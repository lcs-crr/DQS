"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dotenv import dotenv_values
from model_garden.tevae import *
from utilities import data_class

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Declare constants
MODEL_NAME = 'tevae'
SEED = 1

# Set fixed seed for random operations
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load variables in .env file
config = dotenv_values("../.env")

# Load directory paths from .env file
data_path = config['data_path']
model_path = config['model_path']

for fold_idx in range(3):
    for split in ['1day', '1week', '2weeks', '3weeks', '4weeks']:
        data_load_path = os.path.join(data_path, '2_preprocessed', 'fold_' + str(fold_idx), split)
        model_save_path = os.path.join(model_path, MODEL_NAME + '_' + split + '_' + str(fold_idx) + '_1')  # Fixed model seed due to focus on query strategy

        tfdata_train = tf.data.Dataset.load(os.path.join(data_load_path, 'train'))
        tfdata_val = tf.data.Dataset.load(os.path.join(data_load_path, 'val'))

        tfdata_train = tfdata_train.cache().batch(1024).prefetch(tf.data.AUTOTUNE)
        tfdata_val = tfdata_val.cache().batch(1024).prefetch(tf.data.AUTOTUNE)

        # Establish callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_rec_loss',
            mode='min',
            verbose=1,
            patience=250,
            restore_best_weights=True,
        )

        # Define model
        window_size = tfdata_train.element_spec.shape[1]
        features = tfdata_train.element_spec.shape[2]
        with strategy.scope():
            annealing = KLAnnealing(
                annealing_type="cyclical",
                grace_period=25,
            )
            latent_dim = features // 2
            key_dim = features // 8
            hidden_units = features * 16
            encoder = TEVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, features=features,
                                    hidden_units=hidden_units, seed=SEED)
            decoder = TEVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, features=features,
                                    hidden_units=hidden_units, seed=SEED)
            ma = MA(seq_len=window_size, latent_dim=latent_dim, key_dim=key_dim, features=features)
            model = TEVAE(encoder, decoder, ma)
            callback_list = [early_stopping, annealing]
            model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=None))

        # Fit vae model
        history = model.fit(tfdata_train,
                            epochs=10000,
                            callbacks=callback_list,
                            validation_data=tfdata_val,
                            verbose=2
                            )

        # Run model on random data with same shape as input to build model
        model.predict(tf.random.normal((32, window_size, features)), verbose=0)

        # Save model and losses
        os.makedirs(model_save_path, exist_ok=True)
        model.save(os.path.join(model_save_path, 'model.keras'))
        data_class.DataProcessor().dump_pickle(history, os.path.join(model_save_path, 'losses.pkl'))
        tf.keras.backend.clear_session()
