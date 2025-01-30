import pandas as pd
from config_ml import settings, all_features
from src.dataset import GeoDiversityData

include_features = ['NDVI', 'dem', 'slope', 'aspect', 'bio01', 'bio02', 'bio03', 'bio04', 'bio07', 'bio12', 'bio15',
                    'dewpoint', 'max_temp', 'mean_temp', 'min_temp', 'precip', 'volume', 'average_height',
                    'ground_area', 'average_diameter', 'biomass', 'vegetation_quota', 'hii', 'soil_moisture', 'ph']
features = GeoDiversityData.get_feature_list(all_features, include_features=include_features)
diversity_data = GeoDiversityData(pd.read_csv(settings.samples_file), features_list=features)
diversity_data.df.to_csv(settings.data_path / "EXAMPLE_DATA_WITH_NUMERICAL_VALUES.csv", index_label=False)