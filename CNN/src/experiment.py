import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Type, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

from config import settings
from src.dataset import GeoDiversityData, make_kfold_for_geodata, GeoDiversityDatasetPermute,geo_train_test_split,make_training_only_for_geodata
from src.train import train_nn, get_lr_scheduler
from src.metrics import MetricsForDiversity, Logger
from src.eval import eval_nn


def prepare_geo_data(
        include_folders: Optional[list[str]] = None,
        exclude_folders: Optional[list[str]] = None,
        exclude_features: Optional[list[str]] = None,
        additional_features: Optional[list[str]] = None,
        use_spatial_features: Optional[list[str]] = None,
        use_numerical_features: Optional[list[str]] = None,
        prediction: bool = False,

):
    """
    This is a helper function to prepare a GeoDiversityData object
    with the desired features.

    include_folders:
        see GeoDiversityData.get_feature_list
    exclude_folders:
        see GeoDiversityData.get_feature_list
    exclude_features:
        see GeoDiversityData.get_feature_list
    additional_features:
        list of strings, each string STR is used to load a pandas dataframe from
        {settings.cache_path} / STR.parquet
        which is supposed to contain numerical features for the samples in the dataset
        THOSE ARE NORMALIZED BEFORE USE, AND COLUMN NAMES ARE CHANGED TO AVOID CONFLICTS!
    use_spatial_features:
        list of names of spatial features to use
    use_numerical_features:
        list of names of numerical features to use
    """


    numerical_features = GeoDiversityData.get_feature_list(
        spatial=False,
        include_folders=include_folders,
        exclude_folders=exclude_folders,
        exclude_features=exclude_features
    )
    if use_numerical_features:
        numerical_features = [feature for feature in numerical_features if str(feature) in use_numerical_features]

    if use_spatial_features:
        spatial_features = GeoDiversityData.get_feature_list(
            spatial=True,
            include_folders=include_folders,
            exclude_folders=exclude_folders,
            exclude_features=exclude_features
        )
        spatial_features = [feature for feature in spatial_features if str(feature) in use_spatial_features]
    else:
        spatial_features = []

    diversity_data = GeoDiversityData(features_list=numerical_features + spatial_features,use_image_cache=False,prediction=prediction)

    if additional_features is not None:
        for add_emb in additional_features:
            # Load features
            embedding_path = settings.cache_path / (add_emb + '.parquet')
            emb_df = pd.read_parquet(embedding_path)

            # Change column names
            emb_df.columns = [(f"{embedding_path.stem}-{x}" if x != settings.sampleid_col else x) for x in
                              emb_df.columns]

            # Normalize
            # scaler = StandardScaler()
            numeric = emb_df.select_dtypes(include=['number'])
            numeric = (numeric - numeric.mean()) / (numeric.std() + 1e-10)
            for col in numeric.columns:
                emb_df[col] = numeric[col]
            # emb_df.select_dtypes(include=['number'])[:] = numeric

            # Add features
            diversity_data.add_extracted_features(emb_df, temporal=True)

    return diversity_data, numerical_features, spatial_features


def nn_experiment(
        experiment_name: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        scheduler: str,
        ModelType: Type[nn.Module],
        model_params: dict[str],
        weight_decay: float = 0.,
        kfold_random_seed: int = 42,
        kfold_n_splits: int = 7,
        use_folds: Optional[list[int]] = None,
        include_folders: Optional[list[str]] = None,
        exclude_folders: Optional[list[str]] = None,
        exclude_features: Optional[list[str]] = None,
        additional_features: Optional[list[str]] = None,
        use_spatial_features: Optional[list[str]] = None,
        use_numerical_features: Optional[list[str]] = None,
        transforms: Callable = None,
        use_neptune: bool = True,
        tags: Optional[list[str]] = None,
        num_workers: int = settings.num_workers,
        log_dict: Optional[dict[str]] = None,
        train_only: bool = False,
):
    """
    This function performs an experiment, training ModelType using GeoDiversityData.
    It should log all of the choices made in setting this experiment up.

    Many examples of use are in the train_scripts folder.

    experiment_name:
        just for logging using the logger
    batch_size:
        the batch size to use
    num_epochs:
        how many epochs to use for training
    lr:
        the learning rate for training
    scheduler:
        see get_lr_scheduler in src/train.py for available learning rate schedulers, add there if you want another one
    ModelType:
        the class of the model to be loaded
    model_params:
        the parameters to use for initializing ModelType:
            model = ModelType(**model_params)
    weight_decay:
        the weight_decay parameter to use in training (equivalent to L2 regularization)
    kfold_random_seed:
        the random seed used to do the k-fold split of data
    kfold_n_split:
        how many splits to do for cross-val
    use_folds:
        if you want to carry the experiment out only for a subset of folds, put the list of ints here
    include_folders:
    exclude_folders:
    exclude_features:
    additional_features:
    use_spatial_features:
        check prepare_geo_data for documentation
    trainforms:
        torchvision (or other) transforms to be used for spatial data
    use_neptune:
        whether to use neptune.ai for logging
    num_workers:
        how many workers to use;
        WARNING: if spatial data is loaded to memory, it takes quite some time for it to be copied for various workers, which may slow down training;
        moreover, you may easily run out of memory then
    log_dict:
        a dictionary with any additional data that should be logged by the logger
    """

    # Load the data
    diversity_data, numerical_features, spatial_features = prepare_geo_data(
        include_folders=include_folders,
        exclude_folders=exclude_folders,
        exclude_features=exclude_features,
        additional_features=additional_features,
        use_spatial_features=use_spatial_features,
        use_numerical_features=use_numerical_features
    )

    if train_only:
        folds = make_training_only_for_geodata(diversity_data, random_state=kfold_random_seed, shuffle=True)
    else:
        folds = make_kfold_for_geodata(diversity_data, n_splits=kfold_n_splits, shuffle=True, random_state=kfold_random_seed)

    # Set the device
    device = settings.device

    # Set the loss function
    loss_fn = nn.MSELoss()

    # Prepare the object for computing metrics
    metrics_engine = MetricsForDiversity(diversity_data.get_labels().to_numpy() / settings.max_label)

    # Carry out experiments for folds
    for i, fold in enumerate(folds):
        if (use_folds is not None) and (i not in use_folds):
            continue

        if train_only:
            traineval_split = {'train': fold['train']}
        else:
            traineval_split = {'train': fold['train'], 'eval': fold['test']}

        # Prepare datasets and dataloaders
        datasets = diversity_data.get_datasets(
            traineval_split,
            transform_spatial=transforms
        )
        dataloaders = {
            split: DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for split, dataset in datasets.items()
        }

        # Initialize the model
        model_params["n_numeric"] = datasets['train'].num_numeric_features()
        model = ModelType(
            **model_params
        ).to(device)

        # Initialize logging
        logger = Logger(neptune_run=use_neptune, tags=tags)
        save_path = logger.save_path

        # Log all important info
        logger.log_dict(log_dict)

        logger.log_dict({
            "experiment_name": experiment_name,
            "model": str(model),
            "transforms": str(transforms),
        })

        logger.log_dict(
            prefix="params",
            dictionary={
                "lr": lr,
                "epochs": num_epochs,
                "scheduler": scheduler,
                "weight_decay": weight_decay,
            })

        logger.log_dict(
            prefix="data",
            dictionary={
                "fold": i,
                "kfold_random_seed": kfold_random_seed,
                "kfold_n_splits": kfold_n_splits,
                "numerical_features": [str(f) for f in numerical_features],
                "spatial_features": [str(f) for f in spatial_features],
                "additional_features": additional_features,
            })

        logger.log_dict(
            prefix="params/model_params",
            dictionary=model_params
        )

        # Show the model used
        print(model)

        # Set up the optimizer and the learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        lr_scheduler = get_lr_scheduler(scheduler,
                                        optimizer,
                                        num_batches=len(dataloaders["train"]),
                                        three_phase=False,
                                        num_epochs=num_epochs,
                                        base_lr=lr)

        # Train the network!
        model, metric_best = train_nn(
            model=model,
            num_epochs=num_epochs,
            device=device,
            dataloaders=dataloaders,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            save_path=save_path,
            metrics_engine=metrics_engine,
            logger=logger,
            train_only=train_only,
        )


def pfi_experiment(
        model_name: str,
        batch_size: int,
        ModelType: Type[nn.Module],
        model_params: dict[str],
        path_model_weights: str,
        importance_metric: str = 'loss',
        n_repeats: int = 5,
        kfold_random_seed: int = 42,
        kfold_n_splits: int = 7,
        use_splits: list[str] = ["train", "eval"],
        use_folds: Optional[list[int]] = None,
        include_folders: Optional[list[str]] = None,
        exclude_folders: Optional[list[str]] = None,
        exclude_features: Optional[list[str]] = None,
        additional_features: Optional[list[str]] = None,
        use_numerical_features: Optional[list[str]] = None,
        use_spatial_features: Optional[list[str]] = None,
        features_for_pfi: Optional[list[str]] = None,
):

    diversity_data, numerical_features, spatial_features = prepare_geo_data(
        include_folders=include_folders,
        exclude_folders=exclude_folders,
        exclude_features=exclude_features,
        additional_features=additional_features,
        use_spatial_features=use_spatial_features,
        use_numerical_features=use_numerical_features,
    )

    folds = make_kfold_for_geodata(diversity_data, n_splits=kfold_n_splits, shuffle=True, random_state=kfold_random_seed)

    device = settings.device

    loss_fn = nn.MSELoss()

    metrics_engine = MetricsForDiversity(diversity_data.get_labels().to_numpy() / settings.max_label)

    pfis = []
    all_vals = []

    if features_for_pfi is None:
        features_for_pfi = numerical_features + spatial_features

    for i, fold in enumerate(folds):
        if (use_folds is not None) and (i not in use_folds):
            continue

        traineval_split = {'train': fold['train'], 'eval': fold['test']}

        datasets: dict[str, GeoDiversityDatasetPermute] = diversity_data.get_datasets(traineval_split, permutable_dataset=True)

        # Initialize the model
        model_params["n_numeric"] = datasets['train'].num_numeric_features()
        model = ModelType(**model_params).to(device)

        model_weights = torch.load(path_model_weights, map_location=torch.device('cpu'))
        model.load_state_dict(model_weights)


        pfi: dict[str, dict[str, Tuple[float, float]]] = {split: {} for split in use_splits}
        values: dict[str, dict[str, list[float]]] = {split: {} for split in use_splits}

        for split in use_splits:
            for feature in features_for_pfi:
                values[split][feature.name] = []
                for r in range(n_repeats):
                    datasets[split].permute(feature.name)
                    dataloader = DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

                    metrics = eval_nn(
                        model=model,
                        dataloader=dataloader,
                        loss_fn=loss_fn,
                        metrics_engine=metrics_engine,
                        device=device,
                        logger=None,
                    )
                    values[split][feature.name].append(metrics[importance_metric])
                datasets[split].reset()  # Necessary!

                v = values[split][feature.name]
                pfi[split][feature.name] = (np.mean(v), np.std(v))

        pfis.append(pfi)
        all_vals.append(values)

    # Create empty lists to store data
    data_pfis = []

    # Iterate over each element in pfis
    for i, pfi_dict in enumerate(pfis):
        for split, features in pfi_dict.items():
            for feature, stats in features.items():
                data_pfis.append({
                    'fold': i,
                    'split': split,
                    'feature': feature,
                    'mean': stats[0],
                    'std_dev': stats[1]
                })

    # Convert the list of dictionaries to a DataFrame
    df_pfis = pd.DataFrame(data_pfis)

    data_vals = []

    # Iterate over each element in all_vals
    for i, values_dict in enumerate(all_vals):
        for split, features in values_dict.items():
            for feature, values in features.items():
                for value in values:
                    data_vals.append({
                        'fold': i,
                        'split': split,
                        'feature': feature,
                        'value': value
                    })

    # Convert to DataFrame
    df_vals = pd.DataFrame(data_vals)

    # Save the DataFrames to CSV files
    out_dir = settings.root / 'pfi_runs' / model_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_pfis.to_csv(out_dir / 'pfis.csv', index=False)
    df_vals.to_csv(out_dir / 'all_vals.csv', index=False)


def prediction_experiment(
        model_name: str,
        batch_size: int,
        ModelType: Type[nn.Module],
        model_params: dict[str],
        path_model_weights: str,
        include_folders: Optional[list[str]] = None,
        exclude_folders: Optional[list[str]] = None,
        exclude_features: Optional[list[str]] = None,
        additional_features: Optional[list[str]] = None,
        use_numerical_features: Optional[list[str]] = None,
        use_spatial_features: Optional[list[str]] = None,
):
    diversity_data, numerical_features, spatial_features = prepare_geo_data(
        include_folders=include_folders,
        exclude_folders=exclude_folders,
        exclude_features=exclude_features,
        additional_features=additional_features,
        use_spatial_features=use_spatial_features,
        use_numerical_features=use_numerical_features,
        prediction=True,
    )

    device = settings.device

    all_ids = []
    all_predictions = []

    test_trap_ids = diversity_data.get_indices_from_trapids(trap_ids=diversity_data.get_trapids())

    pred_indices = {'pred': test_trap_ids}

    datasets = diversity_data.get_datasets(pred_indices)

    # Initialize the model
    model_params["n_numeric"] = datasets['pred'].num_numeric_features()
    model = ModelType(**model_params).to(device)
    model.eval()

    model_weights = torch.load(path_model_weights, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)
    dataloader = DataLoader(datasets['pred'], batch_size=batch_size, shuffle=False, num_workers=settings.num_workers)

    with torch.no_grad():
        for ids, _, numerical_inputs, spatial_inputs in tqdm(dataloader):
            numerical_inputs = numerical_inputs.to(device)
            spatial_inputs = spatial_inputs.to(device)

            outputs = model(numerical_inputs, spatial_inputs)
            outputs_np = np.clip(outputs.cpu().numpy(), -1e5, 1e5)  # Clip to a large range

            all_ids.extend(ids.cpu().numpy().astype(int))
            all_predictions.extend(outputs_np * settings.max_label)

        # Convert the list of predictions and labels to a DataFrame
        df_predictions = pd.DataFrame({
            'ids': all_ids,
            'predictions': all_predictions,
        })
        df_predictions['predictions'] = df_predictions['predictions'].apply(lambda x: x[0])
        # Save the DataFrame to a CSV file
        current_time = datetime.now().strftime('%y%m%d-%H%M%S')
        out_dir = settings.root / 'prediction_runs' / f"{model_name}_{current_time}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df_predictions.to_csv(out_dir / 'predictions.csv', index=False)
