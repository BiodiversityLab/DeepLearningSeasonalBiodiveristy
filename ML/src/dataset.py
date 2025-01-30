"""
There are several parts to this file.
In the first one, pytorch Datasets are constructed for feeding data into models.
In the second one, GeoDiversityData is defined, which loads and manipulates the data used 
in the project.

Before using this file, read the documentation of config.py,
there is the description of GeoFeature and GeoFeatureFolder.
"""
import pdb
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from typing import Optional, Literal, get_args, Tuple, Callable
from copy import deepcopy
from pathlib import Path
import rasterio
from sklearn.model_selection import train_test_split, KFold
import concurrent.futures
import lzma
import pickle
from PIL import Image
from tqdm import tqdm

from config import settings, all_feature_folders, all_features, GeoFeature, GeoFeatureFolder


class GeoDiversityDataset(Dataset):
    """
    This is a Dataset class used by the pytorch DataLoader 
    to feed data into the models.
    """
    sample_ids: pd.DataFrame

    labels: pd.Series
    features_numeric: pd.DataFrame
    features_spatial: dict[str, list[npt.NDArray]]

    num_samples: int

    transform_label: Optional[Callable]=None
    transform_spatial: Optional[Callable]=None
    transform_numeric: Optional[Callable]=None

    def __init__(self,
        df: pd.DataFrame,
        label_col: str,
        id_col: str,
        features_spatial: dict[str, list[npt.NDArray]],
        transform_label: Optional[Callable]=None,
        transform_spatial: Optional[Callable]=None,
        transform_numeric: Optional[Callable]=None,
    ):
        """
        df:
            a DataFrame containing: ids of samples (for logging purposes), labels of samples, and all numerical features to be used
            (e.g., a subset of df in GeoDiversityData)
        label_col:
            the name of the column that contains labels to be predicted
        id_col:
            the name of the column that contains ids of the samples
        features_spatial:
            a dictionary, with keys strings (currently not used in the code, could refactor)
            and values lists of numpy arrays; each numpy array should be of dimension
            (ch, num, num), where ch is the number of channels and num is the size of the image data;
            currently the dataset just concatenates them to one single spatial feature along axis 0 (channels)
        transform_label:
            transforms to be applied to labels (e.g., could add noise for training)
        transform_spatial:
            transforms to be applied to spatial data (e.g., torchvision transforms) 
            - highly recommended if the CNN/vision part of the model is not frozen!
        transform_numeric:
            transforms to be applied to numerical data (e.g., random noise)
        """
        if len(features_spatial) > 0:
            for key, spatial in features_spatial.items():
                if len(spatial) != len(df):
                    raise ValueError(f"Lengths of `df` and `features_spatial[{key}]` do not match: {len(df)} != {len(spatial)}")
                
        self.sample_ids = df[id_col]
        
        self.labels = df[label_col]
        self.features_numeric = df.drop(columns=[id_col, label_col])
        self.features_spatial = features_spatial

        self.transform_label = transform_label
        self.transform_spatial = transform_spatial
        self.transform_numeric = transform_numeric

        self.num_samples = len(df)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Very straightforward.
        Ouputs are, besides the sample_id, torch tensors containing floats.
        Precisely:
        sample_id:
            the id of the sample, usually a string
        label:
            1d tensor of shape (1,)
        numeric:
            1d tensor of shape (n,)
        spatial:
            3d tensor of shape (ch_all,num,num)
        """
        # Get the sample id
        sample_id = self.sample_ids.iloc[idx]

        # Get the label
        label = self.labels.iloc[idx]
        if self.transform_label is not None:
            label = self.transform_label(label)
        label = torch.tensor(label).unsqueeze(0).float() 

        # Get the numeric features
        numeric = self.features_numeric.iloc[idx]
        if self.transform_numeric is not None:
            numeric = self.transform_numeric(numeric)
        else:
            numeric = numeric.to_numpy()
        numeric = torch.tensor(numeric).float() 

        # Get the spatial features
        spatial = []
        for spatial_list in self.features_spatial.values():
            image = self.load_tiff(spatial_list[idx]) #ADDED 240909 to load from disk instead of memory
            spatial.append(torch.tensor(image).float()) #ADDED 240909
            #spatial.append(torch.tensor(spatial_list[idx]).float()) #original

        if spatial:
            spatial = torch.cat(spatial, dim=0)
            if self.transform_spatial is not None:
                spatial = self.transform_spatial(spatial)
        else:
            spatial = torch.empty((0)).float()

        return sample_id, label, numeric, spatial
    
    def num_numeric_features(self):
        """
        A helper function providing the length of the numeric output vectors.
        """
        return len(self.features_numeric.columns)

    def load_tiff(self, filepath: Path) -> npt.NDArray:
        """
        A helper function to load a single tiff file properly
        (could use imageio instead).
        Returns the tiff converted to a numpy array.
        """
        return rasterio.open(filepath).read()
    

class GeoDiversityDatasetPermute(GeoDiversityDataset):
    """
    This is essentially GeoDiversityDataset,
    but with an option to permute single columns.

    I implemented it for use in PFI, but haven't tested it!
    """
    features_numeric_original: pd.DataFrame
    features_spatial_original: dict[str, list[npt.NDArray]]

    def __init__(self, **kwargs):
        #raise NotImplementedError(f"This class has been implemented but not tested. Test and fix bugs before using it!")
        super().__init__(**kwargs)

        self.features_numeric_original = self.features_numeric
        self.features_spatial_original = self.features_spatial
    
    def reset(self):
        self.features_numeric = self.features_numeric_original.copy()
        self.features_spatial = self.features_spatial_original.copy()

    def permute(self, feature_name: str):
        # Permute a column in the dataframe
        if feature_name in self.features_numeric_original.columns:
            self.features_numeric[feature_name] = np.random.permutation(self.features_numeric_original[feature_name])
        # Or permute a list in the dictionary
        elif feature_name in self.features_spatial_original:
            self.features_spatial[feature_name] = np.random.permutation(self.features_spatial_original[feature_name])
        else:
            raise ValueError(f"Feature {feature_name} not found in features.")



class GeoDiversityData(object):
    """
    This is the primary class to handle data for the project.
    df:
        the pandas DataFrame holds trap ids, sample ids, labels, and numerical features
    num_samples:
        just the length of the df
    features_list:
        this is a list of all features (in GeoFeature format, see config.py) that have been loaded
        to an instance of this class
    image_features:
        this list contains dictionaries of image features;
        elements of this list correspond to consecutive rows of df,
        the length of this list is num_samples
    
    _sampleid_to_trapid:
        dictionary of sample ids to trap ids,
        for use in splitting using trap ids
    """
    df: pd.DataFrame
    num_samples: int
    features_list: list[GeoFeature]
    image_features: list[dict[str,npt.NDArray]]

    _sampleid_to_trapid: dict[str, str]

    def __init__(
            self, 
            df: Optional[pd.DataFrame]=None,
            features_list: Optional[list[GeoFeature]]=None,
            image_features: Optional[list[dict[str, npt.NDArray]]]=None,
            use_image_cache: bool=True,
            save_image_cache: Optional[str|bool]='pickle',
            overwrite_image_cache: bool=False,
            prediction: bool=False,
        ):
        """_summary_

        Args:
            df (Optional[pd.DataFrame], optional): 
                If given, use this df for ids, labels and numeric features. Defaults to None.
            features_list (Optional[list[GeoFeature]], optional): 
                If given, use this list of features intead of all of them. Defaults to None.
            image_features (Optional[list[dict[str, npt.NDArray]]], optional): 
                If given, do not load those an image feature from file, but from this list. Defaults to None.
            use_image_cache (bool, optional): 
                If true, use cached data for loading image features.
                This is faster than loading tiff files.
                Defaults to True.
            save_image_cache (Optional[str | bool], optional): 
                False or string, which could be 'pickle' or 'lzma'. If False, doesn't save image cache.
                If true, saves image cache to pickle/lzma files. 
                Defaults to 'pickle'.
            overwrite_image_cache (bool, optional): 
                If there is a need to overwrite image cache, set this to True. Defaults to False.
        """        

        # Determine the list of features to be used
        if features_list is None:
            self.features_list = self.get_feature_list()
            if image_features is not None:
                spatial_features = [x for x in self.features_list if x.name in image_features[0].keys()]
            else:
                spatial_features = [x for x in self.features_list if x.spatial]
            if df is not None:
                numerical_features = [x for x in self.features_list if x.name in df.columns]
            else:
                numerical_features = [x for x in self.features_list if not x.spatial]
            self.features_list = spatial_features + numerical_features
        else:
            self.features_list = features_list

        # Load csv if needed
        if df is None:
            self.df = pd.read_csv(settings.samples_file)
        else:
            self.df = df
        
        # Set the number of samples
        self.num_samples = len(self.df)

        self.prediction = prediction

        # if not self.prediction:
        #     # Save the sampleid to trapid dictionary
        #     self._sampleid_to_trapid = {}
        #     for _, row in self.df[[settings.trapid_col, settings.sampleid_col]].iterrows():
        #         self._sampleid_to_trapid[row[settings.sampleid_col]] = row[settings.trapid_col]

        # Load features that have been asked for
        self.image_features = [{} for _ in range(self.num_samples)]
        for feature in self.features_list:
            if feature.spatial:
                # The feature may be already given
                if (image_features is not None) and (feature.name in image_features[0]):
                    image_feature_list = [
                        image_dict[feature.name] 
                        for image_dict in image_features
                    ]
                # If it's not, load it
                else:
                    image_feature_list = self.load_image_feature(feature, use_cache=use_image_cache)

                # Now, add it to the list of image features
                for i in range(self.num_samples):
                    self.image_features[i][feature.name] = image_feature_list[i]

                # Finally, cache it 
                if save_image_cache:
                    self.save_cached_feature(feature=feature, format=save_image_cache, overwrite=overwrite_image_cache)
            else:
                if feature.name in self.df.columns: # it's already loaded
                    pass 
                else: # need to load the feature
                    self.df[feature.name] = self.load_numerical_feature(feature)


    def __len__(self) -> int:
        return self.num_samples


    def add_extracted_features(self, df_add: pd.DataFrame, temporal: bool):
        """
        This fuction adds features from df_add DataFrame.
        Currently, the DataFrame is assumed to have the same index as df.
        TODO: change the behavior to merge on settings.sampleid_col.

        Args:
            df_add (pd.DataFrame): 
                The dataframe with features to add. At the moment, should have the same index as df.
            temporal (bool): 
                Whether the feature being added changes over time.
                It's only important for (potentially) filtering features.
        """
        # TODO: make merge work for robustness
        # self.df = self.df.merge(df_add, on=settings.sampleid_col, how='inner')
        self.df = pd.concat([self.df, df_add.drop(columns=[settings.sampleid_col])],
                            axis=1)

        for col in df_add.columns:
            if str(col) != settings.sampleid_col:
                self.features_list.append(GeoFeature(name=col, spatial=False, temporal=temporal, folder='cache'))
    

    def extract_features(self, 
                         feature_extractor: Callable, 
                         feature_name: str,
                         extractor_outputs_list: bool=False) -> pd.DataFrame:
        """
        This generates features from images using a callable feature_extractor.
        Can use this for feature engineering on spatial data.
        For examples of use, check notebook 02.

        Args:
            feature_extractor (Callable): 
                A function that takes as an input a numpy array
                representing a spatial feature, and outputs a float or a list of floats.
            feature_name (str): 
                The name of the feature to process.
            extractor_outputs_list (bool, optional): 
                If the extractor outputs a list of floats, set this to True. Defaults to False.

        Raises:
            ValueError: If the feature is not known.

        Returns:
            pd.DataFrame: 
                A dataframe with the index of self.df, containing the generated features.
        """
        if feature_name not in self.image_features[0]:
            raise ValueError(f"Unknown feature {feature_name}")
        
        extracted_features = []
        print(f"Extracting features from {feature_name}.", file=sys.stderr)
        for i in tqdm(range(self.num_samples), file=sys.stderr):
            features = feature_extractor(self.image_features[i][feature_name])
            extracted_features.append(features)

        if not extractor_outputs_list:
            df_features = pd.DataFrame(extracted_features, index=self.df.index)
        else:
            extracted_features_by_feature_index = []
            for i in range(len(extracted_features[0])):
                extracted_features_by_feature_index.append(
                    [
                        extracted_features_for_image[i]
                        for extracted_features_for_image in extracted_features
                    ]
                )
            df_features = [
                pd.DataFrame(extracted_feature, index=self.df.index)
                for extracted_feature in extracted_features_by_feature_index
            ]

        return df_features


    def _load_images_from_file(self, filepath: Path) -> dict[str, npt.NDArray]:
        """
        A helper function loading image features from a cache file.
        Returns the loaded pickle file.
        Supports .pickle and .xz files.
        """
        if not isinstance(filepath, Path):
            raise ValueError(f"filepath should be a pathlib Path!")
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {str(filepath)}")
        
        if filepath.suffix == '.pickle':
            file = open(filepath, 'rb')
        elif filepath.suffix == '.xz':
            file = lzma.open(filepath, 'rb')
        else:
            raise ValueError(f"File type not supported: {filepath.suffix}")
        
        image_features = pickle.load(file)

        return image_features


    def get_rows_by_indices(self, row_indices):
        """
        A function that generates a subset of GeoDiversityData based on given indices.

        Args:
            row_indices (_type_): 
                List/pd.Series/numpy array of elements of self.df.index indicating the samples
                that should be contained in the returned GeoDiversityData.

        Returns:
            GeoDiversityData: 
                Subset of current GeoDiversityData containing entries
                with indices given by row_indices.
        """
        iloc_indices = [self.df.index.get_loc(row_idx) for row_idx in row_indices]
        image_features = [self.image_features[i].copy() for i in iloc_indices]
        return GeoDiversityData(
            df=self.df.loc[row_indices].copy(),
            features_list=self.features_list.copy(),
            image_features=image_features,
        )

    def divide_by_max_label(x):
        return x / settings.max_label

    def get_datasets(self, 
                     split_indices: dict[str, npt.NDArray], 
                     features: Optional[list[str]]=None,
                     label_col: str = settings.label_col,
                     transform_label: Callable=divide_by_max_label,
                     transform_spatial: Optional[Callable]=None,
                     transform_numeric: Optional[Callable]=None,
                     permutable_dataset: bool=False,
                     ) -> dict[str, GeoDiversityDataset]:
        """
        A function to generate GeoDiversityDatasets from this instance of GeoDiversity Data.
        See src/experiment.py for an example of use.

        Args:
            split_indices (dict[str, npt.NDArray]): 
                A dictionary, the values of which indicate the indices of the elements 
                which should go to corresponding splits.
                The keys are the names of the splits.
            features (Optional[list[str]], optional): 
                Features to use. If None, using all features. If a list, will use only features listed. Defaults to None.
            label_col (str, optional): 
                The name of the column containing the labels. Defaults to settings.label_col.
            transform_label (_type_, optional): 
                The transformation to be applied to labels (passed to the Dataset). Defaults to (lambda x: x / settings.max_label).
            transform_spatial (Optional[Callable], optional): 
                The transformation to be applied to spatial data (passed to the Dataset). Defaults to None.
            transform_numeric (Optional[Callable], optional): 
                The transformation to be applied to numerical data (passed to the Dataset). Defaults to None.
            permutable_dataset (bool, optional): 
                Whether to generate an instance of GeoDiversityDatasetPermute (for use in PFI). Defaults to False.

        Returns:
            dict[str, GeoDiversityDataset]: 
                A dictionary with keys equal to split_indices.keys(),
                and values being the datasets ready to be use with pytorch.
        """
        # A little hack to extract a permutable dataset instead
        if permutable_dataset:
            DatasetClass = GeoDiversityDatasetPermute
        else:
            DatasetClass = GeoDiversityDataset

        splits = split_indices.keys()

        numerical_features_list = self.list_numerical_feature_names()
        if features is not None:
            numerical_features_list = [x for x in numerical_features_list if x in features]

        datasets = {}

        for split in splits:
            data = self.get_rows_by_indices(split_indices[split])
            df = data.df[[settings.sampleid_col, label_col] + numerical_features_list]

            features_spatial = { key: [] for key in data.image_features[0] }
            for spatial_feature in data.image_features:
                for key in features_spatial:
                    features_spatial[key].append(spatial_feature[key])

            datasets[split] = DatasetClass(
                df=df,
                id_col=settings.sampleid_col,
                label_col=label_col,
                features_spatial=features_spatial,
                transform_label=transform_label,
                transform_numeric=transform_numeric,
                transform_spatial=transform_spatial,
            )
        
        return datasets
    

    def get_rows_by_trapids(self, trap_ids: list[str]):
        """A helper function to get GeoDiversityData with entries corresponding to given trap ids.

        Args:
            trap_ids (list[str]): List of trap ids to include.

        Returns:
            GeoDiversityData: A copy of self containing only data from traps with given trapids.
        """
        #indices = self.get_indices_from_trapids(trap_ids=trap_ids)
        return self.get_rows_by_indices(row_indices=trap_ids)


    def get_sampleids(self) -> pd.Series:
        """A helper function returning sample ids.

        Returns:
            pd.Series: Sample ids.
        """
        return self.df[settings.sampleid_col]
    

    def get_indices_from_trapids(self, trap_ids: list[str]) -> pd.Index:
        """A helper function returning all sample indices for given trap ids.

        Args:
            trap_ids (list[str]): A list of trap ids.

        Returns:
            pd.Index: Indices of self.df containing samples from traps with given trap ids.
        """
        return self.df[self.df[settings.trapid_col].isin(trap_ids)].index


    def get_indices_from_sampleids(self, sample_ids: list[str]) -> pd.Index:
        """A helper function returning all sample indices for given trap ids.

        Args:
            trap_ids (list[str]): A list of trap ids.

        Returns:
            pd.Index: Indices of self.df containing samples from traps with given trap ids.
        """
        return self.df[self.df[settings.sampleid_col].isin(sample_ids)].index


    def get_trapids(self) -> npt.NDArray:
        """
        A helper function to get the list of all trap ids.
        """
        return self.df[settings.trapid_col].unique()


    def get_labels(self) -> pd.Series:
        """
        A helper function ot get the list of all labels.
        """
        return self.df[settings.label_col]
    

    def get_numerical_features(self) -> pd.DataFrame:
        """A function returning all of the numerical features.

        Returns:
            pd.DataFrame: a DataFrame containing all of the numerical features.
        """
        return self.df[self.list_numerical_feature_names()]
    

    def get_spatial_features(self) -> list[npt.NDArray]:
        """A function returning all of the spatial features concatenated into numpy arrays.

        Returns:
            list[npt.NDArray]: 
                A list, each element of which contains all of the spatial features
                for the corresponding sample, concatenated into a single numpy array.
        """
        #return [np.concatenate(arrays.values(), axis=0) for arrays in self.image_features]
        return [np.concatenate(list(arrays.values()), axis=0) for arrays in self.image_features]
    

    def get_averaged_spatial_features(self) -> pd.DataFrame:
        """A helper function generating averaged spatial features.

        Returns:
            pd.DataFrame: a dataframe with averages of spatial features.
        """
        df_averages = self.get_transformed_spatial_features(np.average, "avg")
        return df_averages


    def get_transformed_spatial_features(self, transform, suffix="transformed") -> pd.DataFrame:
        """
        Actually, this is almost the same as 'extract_features'.
        I don't remember why I coded two different function for the same job.

        Args:
            transform (_type_): a transform to apply to each spatial feature.
            suffix (str, optional): a suffix to add to the generated columns in the returned dataframe. Defaults to "transformed".

        Returns:
            pd.DataFrame: the DataFrame where each column is the result of transforming a single spatial feature.
        """
        feature_names = self.list_spatial_feature_names()
        df_transforms = pd.DataFrame(index=self.df.index)
        for feature in feature_names:
            series = pd.Series(index=self.df.index, dtype=np.float32)
            for i in range(self.num_samples):
                series.iloc[i] = transform(self.image_features[i][feature])
            df_transforms[f"{feature}_{suffix}"] = series
        return df_transforms


    def load_tiff(self, filepath: Path) -> npt.NDArray:
        """
        A helper function to load a single tiff file properly
        (could use imageio instead).
        Returns the tiff converted to a numpy array.
        """
        return rasterio.open(filepath).read()


    def load_tiffs_path(self, feature: GeoFeature) -> list[npt.NDArray]:
        """
        A helper function that loads images for a single spatial feature
        into a list of numpy arrays.
        Uses settings.num_workers parallel threads for speedup.
        """
        filepaths = []
        for sample_id in self.df[settings.sampleid_col]:
            sample_id = str(sample_id)
            filepaths.append(self.tiff_path(sample_id, feature))

        return filepaths

    def load_tiffs_feature(self, feature: GeoFeature) -> list[npt.NDArray]:
        """
        A helper function that loads images for a single spatial feature
        into a list of numpy arrays.
        Uses settings.num_workers parallel threads for speedup.
        """
        filepaths = []
        for sample_id in self.df[settings.sampleid_col]:
            sample_id = str(sample_id)
            filepaths.append(self.tiff_path(sample_id, feature))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.num_workers) as executor:
            tiffs = executor.map(self.load_tiff, filepaths)
        
        return tiffs
    

    def load_cached_feature(self, feature: GeoFeature) -> list[npt.NDArray]:
        """
        A helper function to load a cached feature from a pickle or xz (lzma-compressed) file.

        Args:
            feature (GeoFeature): the feature to be loaded.

        Raises:
            FileNotFoundError: if no cache is found.

        Returns:
            list[npt.NDArray]: the loaded feature.
        """
        folder_path = settings.cache_path
        pickle_file = folder_path / f"{feature.name}.pickle"
        xz_file = folder_path / f"{feature.name}.xz"
        if pickle_file.exists():
            file = open(pickle_file, "rb")
        elif xz_file.exists():
            file = lzma.open(xz_file, "rb")
        else:
            raise FileNotFoundError(f"Asked to load from cache, but did not find a cache file for {str(feature)}.")
        image_feature = pickle.load(file)
        file.close()
        return image_feature
    

    def save_cached_feature(self, feature: GeoFeature, format: str='pickle', overwrite: bool=False):
        """
        The opposite of load_cached_feature :)

        Args:
            feature (GeoFeature): the feature to be saved.
            format (str, optional): the format to use, either pickle or lzma (xz). Defaults to 'pickle'.
            overwrite (bool, optional): whether to overwrite the existing cache file. Defaults to False.

        Raises:
            ValueError: if the format is not supported.
        """
        folder_path = settings.cache_path
        if format == 'pickle':
            filepath = folder_path / f"{feature.name}.pickle"
            if (not overwrite) and filepath.exists():
                return
            file = open(filepath, "wb")
        elif format == 'lzma' or format == 'xz':
            filepath = folder_path / f"{feature.name}.xz"
            if (not overwrite) and filepath.exists():
                return
            file = lzma.open(filepath, "wb")
        else:
            raise ValueError(f"Compression type {format} not supported.")
        
        print(f"Writing {feature.name} cache to {str(filepath)}")

        image_feature = [ features[feature.name] for features in self.image_features]

        pickle.dump(image_feature, file)
        file.close()


    def load_numerical_feature(self, feature: GeoFeature) -> list[float]:
        """
        This loads a numerical (constant) feature from tiff files.

        Args:
            feature (GeoFeature): the feature to be loaded.

        Returns:
            list[float]: the values of the feature for all samples.
        """
        return [
            arr[0][arr.shape[1]//2][arr.shape[2]//2]
            for arr in self.load_tiffs_feature(feature)
        ]
    

    def load_image_feature(self, 
                           feature: GeoFeature, 
                           use_cache: bool=True
                           ) -> list[npt.NDArray]:
        """
        This loads a chosen spatial feature from tiff files.

        Args:
            feature (GeoFeature): the feature to be loaded.
            use_cache (bool, optional): whether to use cache, if available. Defaults to True.

        Returns:
            list[npt.NDArray]: the loaded feature.
        """
        print(f"Loading {feature.name}", file=sys.stderr)
        # if use_cache: # COMMENTED OUT TO TEST LOADING FROM DISK 240906
        #     try:
        #         image_feature = list(self.load_cached_feature(feature))
        #     except FileNotFoundError:
        #         image_feature = list(self.load_tiffs_feature(feature))
        # else:
        #     image_feature = list(self.load_tiffs_feature(feature))
        image_feature = list(self.load_tiffs_path(feature)) # ADDED 240906

        return image_feature


    def list_features(self,
        spatial: Optional[bool] = None,
        temporal: Optional[bool] = None,
    ) -> list[GeoFeature]:
        """
        Lists all features available, potentially filtering the list.

        Args:
            spatial (Optional[bool], optional): if True or False, lists only spatial or non-spatial features. If None, it lists both. Defaults to None.
            temporal (Optional[bool], optional): if True or False, lists only temporal or non-temporal features. If None, it lists both. Defaults to None.

        Returns:
            list[GeoFeature]: the (filtered) list of features in self.
        """
        return self.get_feature_list(self.features_list, spatial=spatial, temporal=temporal)
    
    
    def list_numerical_feature_names(self) -> list[str]:
        """
        Returns the list of numerical feature names.
        """
        return [str(f) for f in self.list_features(spatial=False)]
    
    def list_spatial_feature_names(self) -> list[str]:
        """
        Returns the list of spatial feature names.
        """
        return [str(f) for f in self.list_features(spatial=True)]


    def tiff_path(self, sample_id: str, feature: GeoFeature) -> Path:
        """
        Generates the path where the tiff file for a specific sample and feature should be found.

        Args:
            sample_id (str): the sample id.
            feature (GeoFeature): the feature.

        Returns:
            Path: the path to the tiff file containing the feature for the sample id.
        """
        folder_path = settings.features_dir / feature.folder
        sample_prefix = ( 
            sample_id 
            # if feature.temporal
            # else self.trap_id(sample_id) + "_all"
        )
        file_name = (
            sample_prefix
            + "-"
            + feature.name 
            + ".tif"
        )
        return folder_path / file_name


    def trap_id(self, sample_id: str) -> str:
        """Returns the trap id corresponding to the sample id."""
        return self._sampleid_to_trapid[sample_id]
    
    @classmethod
    def get_folder_list(
            cls,
            feature_folders: list[GeoFeatureFolder] = all_feature_folders,
            spatial: Optional[bool] = None,
            temporal: Optional[bool] = None,
            include_folders: Optional[list[str]] = None,
            exclude_folders: Optional[list[str]] = None,
            include_features: Optional[list[str]] = None,
            exclude_features: Optional[list[str]] = None,
            ) -> list[GeoFeatureFolder]:
        """
        This filters `feature_folders` (by default, `all_feature_folders`)
        according to various conditions.

        Args:
            feature_folders (list[GeoFeatureFolder], optional): Which feature_folders to filter. Defaults to all_feature_folders.
            spatial (Optional[bool], optional): If set, outputs only spatial or non-spatial (numeric) features in the output folders. Defaults to None.
            temporal (Optional[bool], optional): If set, outputs only temporal or non-temporal features in the output folders. Defaults to None.
            include_folders (Optional[list[str]], optional): If set, uses only the folders listed. Defaults to None.
            exclude_folders (Optional[list[str]], optional): If set, excludes the folders listed. Defaults to None.
            include_features (Optional[list[str]], optional): If set, uses only the features listed. Defaults to None.
            exclude_features (Optional[list[str]], optional): If set, excludes the features listed. Defaults to None.

        Returns:
            list[GeoFeatureFolder]: The filtered list of features.
        """
        folders = []
        for folder in feature_folders:
            included = True
            if spatial is not None:
                included = included and (folder.spatial == spatial)
            if temporal is not None:
                included = included and (folder.temporal == temporal)
            if include_folders is not None:
                included = included and (folder.name in include_folders)
            if exclude_folders is not None:
                included = included and (folder.name not in exclude_folders)
            if not included:
                continue
            
            new_folder = deepcopy(folder)
            if include_features is not None:
                new_folder.features = [
                    x for x in new_folder.features 
                    if x in include_features
                ]
            if exclude_features is not None:
                for feature in exclude_features:
                    new_folder.features.remove(feature)

            folders.append(new_folder)

        return folders

    @classmethod
    def get_feature_list(
            cls,
            features: list[GeoFeature] = all_features,
            spatial: Optional[bool] = None,
            temporal: Optional[bool] = None,
            include_folders: Optional[list[str]] = None,
            exclude_folders: Optional[list[str]] = None,
            include_features: Optional[list[str]] = None,
            exclude_features: Optional[list[str]] = None,
            ) -> list[GeoFeature]:
        """
        This filters `features` (by default, `all_features`)
        according to various conditions.

        Args:
            spatial (Optional[bool], optional): If set, outputs only spatial or non-spatial (numeric) features. Defaults to None.
            temporal (Optional[bool], optional): If set, outputs only temporal or non-temporal features. Defaults to None.
            include_folders (Optional[list[str]], optional): If set, uses only the folders listed. Defaults to None.
            exclude_folders (Optional[list[str]], optional): If set, excludes the folders listed. Defaults to None.
            include_features (Optional[list[str]], optional): If set, uses only the features listed. Defaults to None.
            exclude_features (Optional[list[str]], optional): If set, excludes the features listed. Defaults to None.

        Returns:
            list[GeoFeature]: The filtered list of features.
        """
        return_features = []
        for feature in features:
            included = True
            if spatial is not None:
                included = included and (feature.spatial == spatial)
            if temporal is not None:
                included = included and (feature.temporal == temporal)
            if include_folders is not None:
                included = included and (feature.folder in include_folders)
            if exclude_folders is not None:
                included = included and (feature.folder not in exclude_folders)
            if include_features is not None:
                included = included and (feature.name in include_features)
            if exclude_features is not None:
                included = included and (feature.name not in exclude_features)
            
            if included:
                return_features.append(feature)

        return return_features


"""
Two split strategies were implemented:
either split measurements completely randomly ('simple_random'),
or split by trap location ('spatial').
"""
SplitArgument = Literal['simple_random', 'spatial']


def geo_train_test_split(
        data: GeoDiversityData, 
        split_type: SplitArgument,
        **kwargs  # Args for sklearn's train_test_split
        ) -> Tuple[GeoDiversityData,GeoDiversityData]:
    """
    Uses sklearn's train_test_split to split GeoDiversityData.

    Args:
        data (GeoDiversityData): the data to be split.
        split_type (SplitArgument): 'simple_random' or 'spatial'
        **kwards: passed to sklearn's train_test_split

    Returns:
        Tuple[GeoDiversityData,GeoDiversityData]: train and test data.
    """
    if split_type == 'simple_random':
        train_ids, test_ids = train_test_split(data.df.index, **kwargs)
        return data.get_rows_by_indices(train_ids), data.get_rows_by_indices(test_ids)
    elif split_type == 'spatial':
        all_trap_ids = data.get_trapids()
        train_trap_ids, test_trap_ids = train_test_split(all_trap_ids, **kwargs)
        # train_trap_ids, test_trap_ids = train_test_split(data.get_indices_from_trapids(trap_ids=all_trap_ids), **kwargs)
        #return data.get_rows_by_trapids(train_trap_ids), data.get_rows_by_trapids(test_trap_ids)
        train_trap_ids = data.get_indices_from_trapids(trap_ids=train_trap_ids)
        test_trap_ids = data.get_indices_from_trapids(trap_ids=test_trap_ids)

        return train_trap_ids, test_trap_ids
    else:
        raise ValueError(f"Split type {split_type} unsupported, should be one of {list(get_args(SplitArgument))}")

    


def numpy_to_image(arr: npt.NDArray, factor: float=255., size: Optional[int]=None) -> Image:
    """
    Converts a numpy array to an image, values scaled by factor.
    May be rescaled to (size, size)
    """
    while arr.shape[0] == 1:
        arr = arr[0]

    if factor is not None:
        arr = arr*factor
    
    if len(arr.shape) == 2:
        img = Image.fromarray(np.uint8(arr))
        img = img.convert('RGB')
    else:
        img = Image.fromarray(np.uint8(arr).transpose((1,2,0)), mode="RGB")

    if size is not None:
        img = img.resize((size,size))

    return img

def display_numpy(arr: npt.NDArray):
    """
    A simple function to display a numpy array.
    """
    numpy_to_image(arr).show()


def make_kfold_for_geodata(data: GeoDiversityData, random_state: int=42, **kwargs) -> list[dict[str, GeoDiversityData]]:
    """_summary_

    Args:
        data (GeoDiversityData): the data to split into folds.
        random_state (int, optional): the random seed. Defaults to 42.
        **kwargs: passed to KFold.

    Returns:
        list[dict[str, GeoDiversityData]]: list of folds, each containing 'train' and 'test' splits.
    """
    kf = KFold(random_state=random_state, **kwargs)
    all_trap_ids = data.get_trapids()

    trap_ids = {}
    folds = []
    for train, test in kf.split(all_trap_ids):
        trap_ids["train"], trap_ids["test"] = (all_trap_ids[train], all_trap_ids[test])
        fold_indices = {}
        for split in ["train", "test"]:
            fold_indices[split] = data.get_indices_from_trapids(trap_ids=trap_ids[split])
        folds.append(fold_indices)
    return folds

def make_training_only_for_geodata(data: GeoDiversityData, random_state: int=42, shuffle: bool=True) -> list[dict[str, GeoDiversityData]]:
    np.random.seed(random_state)

    folds = []
    fold_indices = {}

    all_trap_ids = data.get_trapids()
    if shuffle:
        np.random.shuffle(all_trap_ids)
    fold_indices['train'] = data.get_indices_from_trapids(trap_ids=all_trap_ids)

    folds.append(fold_indices)

    return folds
