import sys
import neptune
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from neptune.utils import stringify_unsupported
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer
from dotenv import load_dotenv
import os
from datetime import datetime

from config import settings


class Logger(object):
    """
    This is a class to log train/eval/test info,
    in particular log the training process
    and log the outputs
    and log parameters
    and whatever you wish it to log.

    I used neptune.ai, but it should be easy to customize it for other purposes.

    I did mimick neptune's way of storing data. It logs single values:
    neptune_run["params/lr"] = lr
    or series of values:
    neptune_run["train/loss"].append(loss, step=step)
    and, correspondingly, you can use
    logger.log(name="params/lr", value=lr)
    or
    logger.append(name="train/loss", value=loss, step=step).
    Alternatively, you can log dictionaries easily:
    logger.log_dict(dictionary={"lr":lr}, prefix="params")
    or
    logger.append_dict(dictionary={"loss":loss}, prefix="train", step=step),
    and the dictionaries can be nested.

    Neptune could also have
    neptune_run["train/loss"].append(loss)
    without specifying the step, but I find this approach prone to 
    introducing bugs to the code, so I haven't implemented it.
    """
    epoch: int
    neptune_run: Optional[neptune.Run]
    run_name: str
    tags: list[str]
    outputs: list[pd.DataFrame]
    save_path: Optional[Path]=None

    def __init__(self, 
                 neptune_run: Optional[bool|neptune.Run]=False,
                 run_name: Optional[str]=None,
                 tags: Optional[list[str]] = None,
                 save_path: Optional[Path]=None,
                 save_path_exists_ok: bool=False):
        """
        This initializes the logger.

        neptune_run:
            Either a neptune.Run to be used, or bool indicating whether to use neptune or not.
            If using neptune, should have {settings.root}/.env file with NEPTUNE_PROJECT
            and NEPTUNE_API_TOKEN ready.
        run_name:
            If run_name is not provided, one is generated from the timestamp.
        save_path:
            Where to save outputs.
            If save_path is not specified, outputs are saved in the folder
            {settings.model_path} / {run_name}.
        save_path_exists_ok:
            Whether it's okay that the {save_path} exists.
            Usually we don't want that as we don't want to overwrite previous logs!
        """
        self.epoch = 0
        if isinstance(neptune_run, neptune.Run):
            self.neptune_run = neptune_run
        elif neptune_run:
            self.neptune_run = self._init_neptune(tags=tags)
        else:
            self.neptune_run = None

        if run_name is not None:
            self.run_name = run_name
        elif self.neptune_run is not None:
            self.run_name = self.neptune_run["sys/id"].fetch()
        else:
            current_time = datetime.now()
            self.run_name = current_time.strftime("%Y%m%d_%H-%M-%S")

        if save_path is None:
            self.save_path = settings.model_path / self.run_name
            if (not save_path_exists_ok) and self.save_path.exists():
                raise FileExistsError(f"The path {str(self.save_path)} has already been used for logging!")
            self.save_path.mkdir(parents=True, exist_ok=save_path_exists_ok)
        else:
            self.save_path = save_path

        self.outputs = []

    def __del__(self):
        """
        Properly and gracefully stop.
        """
        self.stop()

    def stop(self):
        """
        Finish logging gracefully and save outputs!
        """
        if self.neptune_run is not None:
            self.neptune_run.stop()

        if self.outputs is not None and self.save_path is not None:
            self.save_outputs(self.save_path)
    
    @classmethod
    def _init_neptune(cls, tags=None):
        """
        Initialize neptune using api token and project name from settings.root / '.env'
        """
        load_dotenv(settings.root / 'neptune_idp.env')
        neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
        neptune_project = os.getenv("NEPTUNE_PROJECT")
        return neptune.init_run(
            project=neptune_project,
            api_token=neptune_api_token,
            tags=tags,
        )

    def set_epoch(self, epoch: int):
        """
        Within the training loop, one may not know which epoch it is,
        so we save it here.
        """
        self.epoch = epoch

    def log(self, name: str, value):
        """
        Logging of a single value.
        """
        if self.neptune_run is not None:
            self.neptune_run[name] = stringify_unsupported(value)

    def log_dict(self, dictionary: dict[str], prefix: str=""):
        """
        Wrapper to log multiple values stored in a dictionary,
        probably a nested one.
        For key, val pair, the val will be stored under f"{prefix}/{key}".
            TODO: probably it's better to have prefix=None by default,
            and then just use f"{key}"!
        """
        if dictionary is None:
            return
        for key, value in dictionary.items():
            name = f"{prefix}/{key}"
            if isinstance(value, dict):
                self.log_dict(prefix=name, dictionary=value)
            else:
                self.log(name, value)

    def append(self, name: str, value, step: float):
        """
        Appending a single value to a series.
        """
        if self.neptune_run is not None:
            self.neptune_run[name].append(value=stringify_unsupported(value), step=step)

    def append_dict(self, dictionary: dict[str], prefix: str=""):
        """
        Wrapper to append multiple values from a dictionary,
        probably a nested one, to corresponding series.
            TODO: probably it's better to have prefix=None by default,
            and then just use f"{key}"!
        """
        for key, value in dictionary.items():
            name = f"{prefix}/{key}"
            if isinstance(value, dict):
                self.append_dict(prefix=name, dictionary=value)
            else:
                self.append(name, value, self.epoch)

    def append_within_epoch(self, name: str, value, epoch_frac: float=0.):
        """
        Wrapper to append a single value to a series without having the epoch provided.
        """
        if self.epoch is None:
            print("ERROR: Unknown epoch, setting epoch=0", file=sys.stderr)
            self.append(name, value, 0. + epoch_frac)
        else:
            self.append(name, value, self.epoch + epoch_frac)
    
    def append_outputs(self, outputs: dict[str], phase: Optional[str]=None):
        """
        Logging of outputs (to be saved to a file)
        """
        outputs = pd.DataFrame(outputs)
        if self.epoch is not None:
            outputs["epoch"] = self.epoch
        if phase is not None:
            outputs["phase"] = phase
        
        self.outputs.append(outputs)

    def save_outputs(self, save_path: Path, save_name: str="outputs"):
        """
        Concatenate self.outputs to a single pandas dataframe
        and self to '{save_path}/{save_name}.parquet'
        """
        # First, we fix problems with mismatched columns, if present
        is_epoch_present = False
        is_phase_present = False

        # Check if we have epoch/phase columns anyway
        for out in self.outputs:
            if "epoch" in out.columns:
                is_epoch_present = True
            if "phase" in out.columns:
                is_phase_present = True

        # Impute values if not everywhere
        for out in self.outputs:
            if is_epoch_present and ("epoch" not in out.columns):
                out["epoch"] = np.nan
            if is_phase_present and ("phase" not in out.columns):
                out["phase"] = "UNKNOWN"

        # Now we try concatenating and saving outputs
        try:
            outputs = pd.concat(self.outputs)
            outputs.to_parquet(save_path / f"{save_name}.parquet")
        except Exception as e:
            print(e)
            print("Concatenating outputs failed, saving to a pickle file.")
            with open(save_path / f"{save_name}.pickle", "wb") as f:
                pickle.dump(self.outputs, f)


def dict_to_metrics_display(dictionary: dict[str]):
    """
    This is a simple function that prepares a dictionary of metrics for nice display.
    """
    metrics_display = [
        f"{key.capitalize()}: {val:.5f}" 
        for key, val in dictionary.items() 
        if isinstance(val, (int, float))
        ]
    return metrics_display


class MetricsForDiversity(object):
    """
    I use this class to generate all the metrics for the models.
    Fit it to your needs.
    At init, it takes the list of labels to be used for fitting the dicretizer
    to later compute 5-class accuracy on classes defined by quantiles.
    At call, it takes labels, preds and, optionally, loss to add it to the dictionary.
    """
    discretizer: KBinsDiscretizer

    def __init__(self, labels: npt.NDArray):
        self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.discretizer.fit(labels.reshape(-1,1))

    def __call__(self, labels: npt.NDArray, preds: npt.NDArray, loss: Optional[float]=None):
        all_metrics = {
            "RMSE" : mean_squared_error(labels, preds, squared=False),
            "R2" : r2_score(labels, preds),
            "MAE" : mean_absolute_error(labels, preds),
            "5class_acc" : accuracy_score(
                self.discretizer.transform(labels.reshape(-1,1)),
                self.discretizer.transform(preds.reshape(-1,1))
            ) 
        }
        if loss is not None:
            all_metrics = {"loss" : loss} | all_metrics
        return all_metrics
