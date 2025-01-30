import sys
sys.path.append(".")
sys.path.append("..")
from pathlib import Path

from src.model import SpatialOnlyResnet18
from src.experiment import prediction_experiment

file_stem = Path(__file__).stem

spatial_features = ['NDVI', 'dem', 'slope', 'aspect', 'bio01', 'bio02', 'bio03', 'bio04', 'bio07', 'bio12', 'bio15',
                    'dewpoint', 'max_temp', 'mean_temp', 'min_temp', 'precip','hii', 'ph', 'soil_moisture',  'volume',
                    'average_height', 'ground_area', 'average_diameter', 'biomass', 'vegetation_quota']

prediction_experiment(
    model_name=file_stem,
    batch_size=32,
    ModelType=SpatialOnlyResnet18,
    model_params={
        "hidden_layers_sizes": [200, 100, 50],
        "dropout": 0.2,
        "dropout_resnet": 0.,
        "dropout_resnet_features": 0.,
        "batch_norm": False,
        "spatial_feature_channels": len(spatial_features),
        "resnet_layers": 5,
        "freeze_resnet": False,
        "pretrained_resnet": False,
    },
    path_model_weights="/path/to/trained/model.pt",
    use_spatial_features=spatial_features,
)
