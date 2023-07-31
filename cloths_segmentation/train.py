import argparse
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from torch.utils.data import DataLoader

from cloths_segmentation.dataloaders import SegmentationDataset
from cloths_segmentation.metrics import binary_mean_iou
from cloths_segmentation.utils import get_samples

image_path = Path(os.environ["IMAGE_PATH"])
mask_path = Path(os.environ["MASK_PATH"])


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class SegmentPeople(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparamsa = hparams

        self.model = object_from_dict(self.hparamsa["model"])
        if "resume_from_checkpoint" in self.hparamsa:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparamsa["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("focal", 0.9, BinaryFocalLoss()),
        ]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        samples = get_samples(image_path, mask_path)

        num_train = int((1 - self.hparamsa["val_split"]) * len(samples))

        self.train_samples = samples[:num_train]
        self.val_samples = samples[num_train:]

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def train_dataloader(self):
        train_aug = from_dict(self.hparamsa["train_aug"])

        if "epoch_length" not in self.hparamsa["train_parameters"]:
            epoch_length = None
        else:
            epoch_length = self.hparamsa["train_parameters"]["epoch_length"]

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, epoch_length),
            batch_size=self.hparamsa["train_parameters"]["batch_size"],
            num_workers=self.hparamsa["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparamsa["val_aug"])

        result = DataLoader(
            SegmentationDataset(self.val_samples, val_aug, length=None),
            batch_size=self.hparamsa["val_parameters"]["batch_size"],
            num_workers=self.hparamsa["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))

        return result

    def configure_optimizers(self):
        print("Model parameters",self.hparamsa["optimizer"])
        optimizer = object_from_dict(
            self.hparamsa["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparamsa["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        total_loss = 0
        logs = {}
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            logs[f"train_mask_{loss_name}"] = ls_mask

        logs["train_loss"] = total_loss

        logs["lr"] = self._get_current_lr()

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        result = {}
        for loss_name, _, loss in self.losses:
            result[f"val_mask_{loss_name}"] = loss(logits, masks)

        result["val_iou"] = binary_mean_iou(logits, masks)

        return result

    def on_validation_epoch_end(self):
        logs = {"epoch": self.trainer.current_epoch}
        return {"log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pipeline = SegmentPeople(hparams)

    Path(hparams["checkpoint_callback"]["dirpath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=WandbLogger(hparams["experiment_name"]),
        callbacks=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
