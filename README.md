# The utility of deep learning for modeling seasonal biodiversity changes across the landscape

Presented here is the bulk code for processing, training, and predicting biodiversity data using machine learning.

## Directories and files
- `CNN`: Contains scripts for the CNN apporach.
  - `CNN/model_scripts`: Scripts for training and predicting CNN models
      - `CNN/model_scripts/01_resnet_fc_final.py`: This script contains the final model setup.
      - `CNN/model_scripts/02_resnet_fc_final_predictions.py`: This script runs predictions based on a model.
  - `CNN/scripts`: Various utility scripts.
    - `CNN/scripts/generate_image_cache.py`: Generates cache for spatial data.
    - `CNN/scripts/preprocessing.py`: Preprocesses data the IBA biodiveristy data.
    - `CNN/scripts/get_numerical_features.py`: Extracts the numerical values from images (used for other ML approaches).
  - `CNN/src`: Key modules used for carrying out experiments in the project.
      - `CNN/src/dataset.py`: Contains classes and functions for loading and preprocessing datasets.
      - `CNN/src/eval.py`: Contains functions for evaluating model performance.
      - `CNN/src/experiment.py`: Contains functions for setting up and running experiments.
      - `CNN/src/metrics.py`: Contains functions for logging and calculating performance metrics.
      - `CNN/src/model.py`: Contains the definition of the ResNet-18 model and other model architectures.
      - `CNN/src/train.py`: Contains functions for training models.
  - `CNN/config.py`: Project-wide settings.
  - `CNN/requirements.txt`: Packages required for the project.

- `ML`: Contains scripts for the other machine learning models.
  - `ML/src`: Key modules used for carrying out experiments in the project.
    - `ML/src/dataset.py`: Contains classes and functions for loading and preprocessing datasets.
    - `ML/src/ml_script.py`: Contains training, evaluation, and prediction functions for the ML models.
  - `ML/config.py`: Project-wide settings.