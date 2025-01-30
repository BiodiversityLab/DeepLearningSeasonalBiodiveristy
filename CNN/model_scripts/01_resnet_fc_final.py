import sys
sys.path.append(".")
sys.path.append("..")
from pathlib import Path
from torchvision.transforms import v2
import argparse

from src.model import SpatialOnlyResnet18
from src.experiment import nn_experiment


def main(args):
    file_stem = Path(__file__).stem

    spatial_features = ['NDVI', 'dem', 'slope', 'aspect', 'bio01', 'bio02', 'bio03', 'bio04', 'bio07', 'bio12', 'bio15',
                        'dewpoint', 'max_temp', 'mean_temp', 'min_temp', 'precip','hii', 'ph', 'soil_moisture',
                        'volume', 'average_height', 'ground_area', 'average_diameter', 'biomass', 'vegetation_quota']
    degrees = 180
    translate_xy = 0.1
    scale_bottom = 0.9
    scale_top = 2.78
    shear = 7.
    fill = 0.
    interpolation = v2.InterpolationMode.BILINEAR

    transforms = v2.RandomAffine(
        degrees=degrees,
        translate=(translate_xy,translate_xy),
        scale=(scale_bottom,scale_top),
        shear=(shear,shear,shear,shear),
        interpolation=interpolation,
        fill=fill,
        )

    nn_experiment(
        experiment_name=file_stem,
        batch_size=args.batch_size,
        num_epochs=100,
        lr=args.lr,
        weight_decay=2e-4,
        scheduler='onecycle', #None, 'onecycle' or 'step'
        ModelType=SpatialOnlyResnet18,
        model_params={
            "hidden_layers_sizes": [200, 100, 50],
            "dropout": args.dropout,
            "dropout_resnet": 0.,
            "dropout_resnet_features": 0.,
            "batch_norm": False,
            "spatial_feature_channels": len(spatial_features),
            "resnet_layers": 5,
            "freeze_resnet": False,
            "pretrained_resnet": False,
        },
        use_spatial_features=spatial_features,
        use_neptune=True,
        transforms=transforms,
        log_dict={
            "data/transform/degrees": degrees,
            "data/transform/translate_xy": translate_xy,
            "data/transform/scale_bottom": scale_bottom,
            "data/transform/scale_top": scale_top,
            "data/transform/shear": shear,
            "data/transform/interpolation": str(interpolation),
            "data/transform/fill": fill,
        },
        train_only=False, # Set to True to train on the full dataset
        kfold_n_splits=7, # Set to 0 to train on the full dataset, otherwise set to the number of folds
        tags=[args.tags],
        num_workers=4,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int)
    parser.add_argument('--lr', action='store', type=float)
    parser.add_argument('--dropout', action='store', type=float)
    parser.add_argument('--tags', action='store', type=str)
    args = parser.parse_args()

    main(args)
