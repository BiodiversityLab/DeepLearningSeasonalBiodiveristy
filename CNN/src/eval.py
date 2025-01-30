import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import settings

from src.metrics import Logger, MetricsForDiversity
from src.train import main_loop


def eval_nn(
        model: nn.Module,
        dataloader: dict[str, DataLoader],
        loss_fn: nn.Module,
        metrics_engine: MetricsForDiversity,
        device: torch.device = settings.device,
        logger: Logger = None,
):
    since = time.time()

    if logger is not None:
        logger.set_epoch(0)

    metrics = main_loop(
        phase="eval_fn",
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        device=device,
        optimizer=None,
        scheduler=None,
        metrics_engine=metrics_engine,
        logger=logger
    )

    time_elapsed = time.time() - since

    print(f"Eval complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")

    return metrics
