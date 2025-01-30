import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import time

from config import settings

from src.model import (
    load_torch_state_dict,
    save_torch_state_dict,
)
from src.metrics import (
    Logger,
    dict_to_metrics_display,
    MetricsForDiversity,
)


def get_lr_scheduler(scheduler,
                     optimizer,
                     num_batches: int,
                     three_phase: bool = True,
                     num_epochs: Optional[int] = None,
                     base_lr: Optional[float] = None):
    if scheduler is None:
        lr_scheduler = None

    elif scheduler.lower() == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * num_batches, gamma=0.94)

    elif scheduler.lower() == "onecycle":
        if num_epochs is None or base_lr is None:
            raise ValueError("Please provide num_epochs and base_lr for the onecycle scheduler.")
        num_epochs = num_epochs
        pct_start = 0.3
        max_lr = base_lr

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=num_epochs * num_batches,
                                                           max_lr=max_lr, pct_start=pct_start, three_phase=three_phase)
    else:
        raise NotImplementedError(f"No learning rate scheduler found for input: {scheduler}")

    return lr_scheduler


def main_loop(
        phase: str,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        metrics_engine: MetricsForDiversity = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        gradient_accumulation_steps: int = 1,
        logger: Logger = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate a single training/validation/test phase and return predictions and labels.
    """

    if phase == "train":
        model.train()  # Set model to training mode
    else:
        model.eval()  # Set model to evaluate mode

    all_ids = []  # For logging purposes
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0
    gradients_accumulated = 0

    steps = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=steps, desc=f"{phase.capitalize()}: ")  # Add a progress bar

    # Iterate over data.
    for step, (ids, labels, numerical_inputs, spatial_inputs) in pbar:
        labels = labels.to(device)
        numerical_inputs = numerical_inputs.to(device)
        spatial_inputs = spatial_inputs.to(device)

        # zero the parameter gradients
        if (phase == "train") and (gradients_accumulated == 0):
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(numerical_inputs, spatial_inputs)

            loss = loss_fn(outputs, labels)

            # backward + optimize only if in training phase
            if phase == "train":
                loss.backward()

                gradients_accumulated += 1
                if gradients_accumulated == gradient_accumulation_steps:
                    optimizer.step()
                    gradients_accumulated = 0

                if scheduler is not None:
                    scheduler.step()

        all_ids.extend(ids)
        all_labels.append(labels.cpu().detach().numpy().flatten())
        all_preds.append(outputs.cpu().detach().numpy().flatten())

        total_loss += loss.item() * numerical_inputs.size(0)
        total_samples += numerical_inputs.size(0)

        running_loss = total_loss / total_samples

        postfix = {"running_loss": running_loss}

        if phase == "train":
            if scheduler is not None:
                postfix["lr"] = scheduler._last_lr[0]

            if logger is not None:
                epoch_frac = (step + 1) / steps
                logger.append_within_epoch(name="train/running_loss", value=running_loss, epoch_frac=epoch_frac)

                if scheduler is not None:
                    logger.append_within_epoch(name="train/learning_rate", value=scheduler._last_lr[0],
                                               epoch_frac=epoch_frac)

        pbar.set_postfix(**postfix)

    if (phase == "train") and (gradients_accumulated > 0):
        optimizer.step()

    all_ids = np.asarray(all_ids)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    if metrics_engine is not None:
        metrics = metrics_engine(loss=running_loss, labels=all_labels, preds=all_preds)
    else:
        metrics = {
            "loss": running_loss
        }

    metrics_display = dict_to_metrics_display(metrics)
    print(phase.capitalize(), *metrics_display)

    if logger is not None:
        logger.append_dict(prefix=phase, dictionary=metrics)
        logger.append_outputs(
            {
                "ids": all_ids,
                "labels": all_labels,
                "preds": all_preds
            },
            phase=phase
        )

    return metrics


def train_nn(
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_path: Path,
        metrics_engine: MetricsForDiversity,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        device: torch.device = settings.device,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 25,
        eval_epochs: int = 1,
        model_name: str = "best_model",
        metric_key: str = "loss",
        metric_best: float = -1.,  # -1 to ensure sth gets saved
        logger: Logger = None,
        test_at_zeroth_epoch: bool = False,
        eval_at_zeroth_epoch: bool = False,
        save_all_checkpoints: bool = False,
        train_only: bool = False,
):
    """
    The training function, mostly standard.
    Logging:
    - shows a progress bar,
    - saves the best model as evaluated on val accuracy,
    - saves neptune info (loss and accuracy at each step and phase),
    - computes the total time spent on training.
    """
    settings.init_seeds()

    since = time.time()

    """
    We have epochs from 0 to num_epochs inclusive:
        - epoch 0 is only for eval
        - epochs 1 to num_epochs are for train and eval
    """
    epoch_best = 0

    if train_only:
        all_phases = ["train"]
    else:
        all_phases = ["train", "eval"]

    for epoch in range(num_epochs + 1):
        print("#" * 40)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        # Makes logging easier
        if logger is not None:
            logger.set_epoch(epoch)

        if epoch == 0:
            phases = []
            if eval_at_zeroth_epoch:
                phases = ["eval"] + phases
            if test_at_zeroth_epoch:
                phases = ["test"] + phases
        elif epoch % eval_epochs == 0 or epoch == num_epochs:
            phases = all_phases
        else:
            phases = ["train"]

        if epoch == 1:
            since_train = time.time()

        for phase in phases:
            metrics = main_loop(
                phase=phase,
                dataloader=dataloaders[phase],
                model=model,
                loss_fn=loss_fn,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                metrics_engine=metrics_engine,
                logger=logger
            )

            # Save the model
            if phase == "eval" and (metric_best < 0 or metrics[metric_key] < metric_best):
                epoch_best = epoch
                metric_best = metrics[metric_key]
                save_torch_state_dict(model, save_path / model_name)
                # if logger is not None:
                #     logger.log_dict(prefix="best_eval", dictionary=metrics)

            if save_all_checkpoints:
                save_torch_state_dict(model, save_path / f"checkpoint{epoch}")

        if epoch > 0:
            time_per_epoch = (time.time() - since_train) / epoch
            eta = (num_epochs - epoch) * time_per_epoch
            eta_h, eta_rem = divmod(eta, 60 ** 2)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_s = int(eta_s)
            print(f"Train speed: {time_per_epoch:.2f}s/epoch",
                  f"Estimated time remaining: {eta_h}h:{eta_m}m:{eta_s}s\n",
                  sep=" | ")
    save_torch_state_dict(model, save_path / "last_epoch")
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    if not train_only:
        print(f"Best val {metric_key.capitalize()} (epoch {epoch_best}): {metric_best * 100:.2f}\n", sep="\n")

    # load best/last model weights, evaluate on test set
    model = load_torch_state_dict(model, save_path / "last_epoch")
    if logger is not None:
        logger.set_epoch(epoch_best)
        logger.log(name="last_epoch", value=num_epochs)

    if "test" in dataloaders:
        test_metrics = main_loop(
            phase="test",
            dataloader=dataloaders["test"],
            model=model,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            metrics_engine=metrics_engine,
            logger=logger
        )
    elif "eval" in dataloaders:
        best_eval_metrics = main_loop(
            phase="best_eval",
            dataloader=dataloaders["eval"],
            model=model,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            metrics_engine=metrics_engine,
            logger=logger
        )

    return model, metric_best