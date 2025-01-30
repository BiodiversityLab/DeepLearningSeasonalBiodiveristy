"""
This file is meant to hold configuration used in various parts of the project.
The main class holding that information is the Settings class,
an instance of which is initialized here as settings.
"""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import random

month = 'june'
    
@dataclass
class Settings(object):
    # The root directory of the project
    root: Path = Path(__file__).parent.resolve()

    # The path where data is stored
    data_path: Path = root / 'data'

    # The path where individual features are stored
    features_dir: Path = root / 'features'

    # The path to where to save predictions
    prediction_dir: Path = data_path / 'prediction' / month

    # The file containing the prediction data
    prediction_file: Path = data_path / 'MODEL_predictions_DATE-TIME.csv'

    # The file with the list of samples and numerical features
    samples_file: Path = data_path / 'SAMPLES.csv'

    # The name of column in samples_file containing trap ids
    trapid_col: str = 'trapID'

    # The name of column in samples_file containing sample ids
    sampleid_col: str = 'unique_id'

    # The name of column in samples_file containing the label to be used across the project
    label_col: str = 'n_taxa'

    # The maximum label, to be used for scaling the labels if needed
    max_label: int = 1189

    # The number of workers to be used for parallel tasks. Should not be bigger than # of CPUs available.
    num_workers: int = 4 # for parallelization

    # The default random seed to be used
    seed: int = 42


    def init_seeds(self):
        """
        A helper function to ensure high reproducibility of the code,
        i.e., that every time the code is ran it produces the same results.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.mps.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = True # Could set for full reproducibility, but leads to speed decrease


# Use this to access the variables described above
settings = Settings()


"""
Here I define helper classes that I used to explicitly define data to be used in the project.
"""
@dataclass
class GeoFeatureFolder(object):
    """
    This describes a folder under settings.features_dir containing tiff files,
    where name is the name of the folder,
    spatial is True if and only if the feature is non-constant,
    temporal is True if and only if the feature changes in time for a specific trap.
    """
    name: str
    spatial: bool
    temporal: bool
    features: list[str]

    def __str__(self):
        return self.name

@dataclass
class GeoFeature(object):
    """
    This describes a single feature and its qualities.
    Useful for loading data, filtering features to be used, etc.
    """
    name: str
    folder: str
    spatial: bool
    temporal: bool

    def __str__(self):
        return self.name

all_feature_folders = [
    GeoFeatureFolder(
        name='days_up',
        spatial=False,
        temporal=True,
        features=['days-up'],
    ),
    GeoFeatureFolder(
        name='dem',
        spatial=False,
        temporal=False,
        features=['dem', 'slope', 'aspect'],
    ),
    GeoFeatureFolder(
        name='elevation_gradient',
        spatial=False,
        temporal=False,
        features=['elevation_gradient'],
    ),
    GeoFeatureFolder(
        name='bioclim',
        spatial=False,
        temporal=False,
        features=[f"bio{i:02}" for i in range(1, 20)],
    ),
    GeoFeatureFolder(
        name='era5_daily',#+'_'+month,
        spatial=False,
        temporal=True,
        features=['dewpoint', 'max_temp', 'mean_temp', 'min_temp', 'precip'],
    ),
    GeoFeatureFolder(
        name='forest_attributes',
        spatial=False,
        temporal=False,
        features=['volume', 'average_height', 'ground_area', 'average_diameter', 'biomass', 'p95', 'vegetation_quota', 'unixdate', 'leaves_presence', 'ground_cover_class']
    ),
    GeoFeatureFolder(
        name='hii',
        spatial=False,
        temporal=False,
        features=['hii', 'infrastructure', 'land_use', 'population_density', 'power', 'railways', 'roads', 'water']
    ),
    GeoFeatureFolder(
        name='soil_moisture',
        spatial=False,
        temporal=False,
        features=['soil_moisture'],
    ),
    GeoFeatureFolder(
        name='s2_monthly',#+'_'+month,
        spatial=False,
        temporal=True,
        features=['NDVI'],
    ),
    GeoFeatureFolder(
        name='ph',
        spatial=False,
        temporal=False,
        features=['ph'],
    ),
]

def get_features_from_folders(feature_folders: list[GeoFeatureFolder]):
    """
    This is a helper function that converts a list of folders into a list of features (contained in them).
    """
    return [
        GeoFeature(
            name=name, 
            folder=feature_folder.name, 
            spatial=feature_folder.spatial, 
            temporal=feature_folder.temporal
        )
        for feature_folder in feature_folders
        for name in feature_folder.features
    ]

def get_folders_from_features(features: list[GeoFeature]):
    """
    This is a helper function that converts a list of features into a list of folders (containing them).
    """
    folder_names = np.unique(np.array(
        [feature.folder for feature in features]
    ))
    feature_folders = []
    for folder_name in folder_names:
        spatial = None
        temporal = None
        feature_list = []
        for feature in features:
            if feature.folder == folder_name:
                spatial = feature.spatial
                temporal = feature.temporal
                feature_list.append(feature)
        folder = GeoFeatureFolder(
            name=folder_name,
            spatial=spatial,
            temporal=temporal,
            features=feature_list,
        )
        feature_folders.append(folder)
    return feature_folders

"""
And this is the list of all features available for the project.
"""
all_features = get_features_from_folders(all_feature_folders)
