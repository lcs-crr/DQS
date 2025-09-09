# DQS: A Low-Budget Query Strategy for Enhancing Unsupervised Data-driven Anomaly Detection Approaches 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work integrates active learning with an existing unsupervised anomaly detection method by selectively querying the labels of multivariate time series, which are then used to refine the threshold selection process.
To achieve this, we introduce a novel query strategy called the dissimilarity-based query strategy (DQS).
DQS aims to maximise the diversity of queried samples by evaluating the similarity between anomaly scores using dynamic time warping.
The preprint of the paper corresponding to this repository can be found on [arXiv](https://arxiv.org/abs/2509.05663).

## Data Set Download
If you are interested in working with the dataset used, please refer to the corresponding [PATH](https://github.com/lcs-crr/PATH) repository 

## Reproducing Results 
Working scripts can be found in the `src` folder: 
- `1_data.py` performs data processing prior to training (downsampling, standardising, windowing, converting to tf.data).
- `2_training.py` performs model training.
- `3_inference.py` does the inference on the validation and testing subsets.
- `4_evaluation_<method>.py` evaluates the results from inference using different methods and query strategies.
  
The source code for `TeVAE` can be found in the `model_garden` folder.

The remaining scripts can be executed in that order to obtain the results in the paper.

Utility functions can be found in the `utilities` folder in this repository. It contains all the classes and the corresponding methods used throughout this work.

Custom model classes for each of the tested approaches can be found in the `model_garden` folder in this repository.

Typically, a `.env` file should be excluded from version control, though we have added a dummy one (`.env_dummy`) to illustrate the file structure.

`requirements.txt` (venv) and `pyprojects.toml` (uv) contain all libraries used.

## Questions?
If any questions or doubts persist, feel free to contact `Lucas Correia` via [Email](mailto:l.ferreira.correia@liacs.leidenuniv.nl) or [LinkedIn](https://www.linkedin.com/in/lcs-crr/).
