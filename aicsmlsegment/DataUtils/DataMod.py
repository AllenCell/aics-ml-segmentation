from aicsmlsegment.DataUtils.Universal_Loader import (
    UniversalDataset,
    RNDTestLoad,
)
import random
from glob import glob
from torch.utils.data import DataLoader
import pytorch_lightning
from aicsmlsegment.Model import get_loss_criterion
import numpy as np


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config
        self.model_name = config["model"]["name"]

        if train:
            self.loader_config = config["loader"]
            self.model_config = config["model"]

            name = config["loader"]["name"]
            if name not in ["default", "focus"]:
                print("other loaders are under construction")
                quit()
            if name == "focus":
                self.check_crop = True
            else:
                self.check_crop = False
            self.transforms = []
            if "Transforms" in self.loader_config:
                self.transforms = self.loader_config["Transforms"]

            _, self.accepts_costmap = get_loss_criterion(config)

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":  # no setup is required for testing
            # load settings #
            config = self.config

            # get validation and training filenames from input dir from config
            validation_config = config["validation"]
            loader_config = config["loader"]
            if validation_config["metric"] is not None:
                print("Preparing train/validation split...", end=" ")
                filenames = glob(loader_config["datafolder"] + "/*_GT.ome.tif")
                filenames.sort()
                total_num = len(filenames)
                LeaveOut = validation_config["leaveout"]
                if len(LeaveOut) == 1:
                    if LeaveOut[0] > 0 and LeaveOut[0] < 1:
                        num_train = int(np.floor((1 - LeaveOut[0]) * total_num))
                        shuffled_idx = np.arange(total_num)
                        random.shuffle(shuffled_idx)
                        train_idx = shuffled_idx[:num_train]
                        valid_idx = shuffled_idx[num_train:]
                    else:
                        valid_idx = [int(LeaveOut[0])]
                        train_idx = list(
                            set(range(total_num)) - set(map(int, LeaveOut))
                        )
                elif LeaveOut:
                    valid_idx = list(map(int, LeaveOut))
                    train_idx = list(set(range(total_num)) - set(valid_idx))
                valid_filenames = []
                train_filenames = []
                # remove file extensions from filenames
                for fi, fn in enumerate(valid_idx):
                    valid_filenames.append(filenames[fn][:-11])
                for fi, fn in enumerate(train_idx):
                    train_filenames.append(filenames[fn][:-11])

                self.valid_filenames = valid_filenames
                self.train_filenames = train_filenames
                print("Done.")

            else:
                print("need validation in config file")
                quit()

    def train_dataloader(self):
        loader_config = self.loader_config
        model_config = self.model_config

        if "unet_xy" in self.model_name:
            size_in = model_config["size_in"]
            size_out = model_config["size_out"]
            nchannel = model_config["nchannel"]

        else:
            size_in = model_config["patch_size"]
            size_out = size_in
            nchannel = model_config["in_channels"]

        train_set_loader = DataLoader(
            UniversalDataset(
                self.train_filenames,
                loader_config["PatchPerBuffer"],
                size_in,
                size_out,
                nchannel,
                use_costmap=self.accepts_costmap,
                transforms=self.transforms,
                patchize=True,
                check_crop=self.check_crop,
                init_only=True,
            ),
            batch_size=loader_config["batch_size"],
            shuffle=True,
            num_workers=loader_config["NumWorkers"],
            pin_memory=True,
        )
        return train_set_loader

    def val_dataloader(self):
        loader_config = self.loader_config
        model_config = self.model_config

        if "unet_xy" in self.model_name:
            size_in = model_config["size_in"]
            size_out = model_config["size_out"]
            nchannel = model_config["nchannel"]

        else:
            size_in = model_config["patch_size"]
            size_out = size_in
            nchannel = model_config["in_channels"]

        val_set_loader = DataLoader(
            UniversalDataset(
                self.valid_filenames,
                loader_config["PatchPerBuffer"],
                size_in,
                size_out,
                nchannel,
                transforms=[],  # no transforms for validation data
                use_costmap=self.accepts_costmap,
                patchize=False,  # validate on entire image
            ),
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=loader_config["NumWorkers"],
            pin_memory=True,
        )
        return val_set_loader

    def test_dataloader(self):
        test_set_loader = DataLoader(
            RNDTestLoad(self.config),
            batch_size=1,
            shuffle=False,
            num_workers=self.config["NumWorkers"],
            pin_memory=True,
        )
        return test_set_loader
