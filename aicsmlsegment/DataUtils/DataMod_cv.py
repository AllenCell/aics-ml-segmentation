from aicsmlsegment.DataUtils.Universal_Loader import (
    UniversalDataset,
    TestDataset,
    load_img,
)
import random
from glob import glob
from torch.utils.data import DataLoader
import pytorch_lightning
from aicsmlsegment.Model import get_loss_criterion
import numpy as np
import torch
from math import ceil
from typing import Dict
from sklearn.model_selection import KFold


def init_worker(worker_id: int):
    """
    Divides the testing images equally among workers

    Parameters
    ----------
    worker_id: int
        id of worker, used to assign start and end images

    Return: None
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # divide images among all workers
    per_worker = int(ceil(len(dataset.filenames) / float(worker_info.num_workers)))
    dataset.start = worker_info.id * per_worker
    dataset.end = min(len(dataset.filenames), (worker_info.id + 1) * per_worker) - 1

# cross validation version of DataModule, just don't want to corrupt the original DataModule file.
class DataModule_CV(pytorch_lightning.LightningDataModule):
    def __init__(self, config: Dict, train: bool = True):
        """
        Initialize Datamodule variable based on config

        Parameters
        ----------
        config: Dict
            a top level configuration object describing which images to load,
            how to load them, and what transforms to apply

        Return: None
        """
        super().__init__()
        self.config = config

        try:  # monai
            self.nchannel = self.config["model"]["nchannel"]
        except KeyError:  # custom model
            self.nchannel = self.config["model"]["in_channels"]

        if train:
            self.loader_config = config["loader"]

            name = self.loader_config["name"]
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

            self.init_only = False
            if self.loader_config["epoch_shuffle"] is not None:
                self.init_only = True

            model_config = config["model"]
            if "unet_xy" in config["model"]["name"]:
                self.size_in = model_config["size_in"]
                self.size_out = model_config["size_out"]
                self.nchannel = model_config["nchannel"]

            else:
                self.size_in = model_config["patch_size"]
                self.size_out = self.size_in
                self.nchannel = model_config["in_channels"]
        else:
            loader_config = config["mode"]
             # during testing, we need to get the same validation data to get their predictions.
            if type(loader_config["InputDir"]) == str:
                loader_config["InputDir"] = [loader_config["InputDir"]]

            filenames = []
            for folder in loader_config["InputDir"]:
                fns = glob(folder + "/*_GT.ome.tif")
                fns.sort()
                filenames += fns

            total_num = len(filenames)
            all_idx = np.arange(total_num)
            kf = KFold(n_splits=config["total_cross_vali_num"], shuffle=True, random_state=12345)
            splits = kf.split(all_idx)
            for i, (this_train_idx, this_test_idx) in enumerate(splits):
                train_idx = this_train_idx
                valid_idx = this_test_idx
                if i == config["current_cross_vali_fold"]:
                    break
                
            valid_filenames = []
            train_filenames = []
            # remove file extensions from filenames
            for fi, fn in enumerate(valid_idx):
                valid_filenames.append(filenames[fn][:-11])
            for fi, fn in enumerate(train_idx):
                train_filenames.append(filenames[fn][:-11])

            self.valid_filenames = valid_filenames
            self.train_filenames = train_filenames
            # we should copy the image and ground truth to the output dir if it does not exist
            import shutil
            import pathlib
            import os
            for fn in self.valid_filenames:
                file_name_stem = pathlib.PurePath(fn).stem
                if not os.path.exists(config['OutputDir']+os.sep+file_name_stem+'.ome.tif'):
                    shutil.copy(fn+'.ome.tif', config['OutputDir']+os.sep+file_name_stem+'.ome.tif')
                if not os.path.exists(config['OutputDir']+os.sep+file_name_stem+'_GT.ome.tif'):
                    shutil.copy(fn+'_GT.ome.tif', config['OutputDir']+os.sep+file_name_stem+'_GT.ome.tif')
                

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        """
        Set up identical train/val splits across gpus. Since all image in batches must
        be the same size, if random splits are selected in the config, the loader will
        try 10 random splits until all of the validation images are the same size.

        Parameters
        ----------
        stage: str
            either "fit" or not

        Return: None
        """
        print('in setup')
        if stage == "fit":
            # load settings #
            config = self.config

            # get validation and training filenames from input dir from config
            validation_config = config["validation"]
            loader_config = config["loader"]
            if validation_config["metric"] is not None:

                if type(loader_config["datafolder"]) == str:
                    loader_config["datafolder"] = [loader_config["datafolder"]]

                filenames = []
                for folder in loader_config["datafolder"]:
                    fns = glob(folder + "/*_GT.ome.tif")
                    fns.sort()
                    filenames += fns

                total_num = len(filenames)
                LeaveOut = validation_config["leaveout"]

                all_idx = np.arange(total_num)
                kf = KFold(n_splits=config["total_cross_vali_num"], shuffle=True, random_state=12345)
                splits = kf.split(all_idx)
                for i, (this_train_idx, this_test_idx) in enumerate(splits):
                    train_idx = this_train_idx
                    valid_idx = this_test_idx
                    if i == config["current_cross_vali_fold"]:
                        break
                valid_filenames = []
                train_filenames = []
                # remove file extensions from filenames
                for fi, fn in enumerate(valid_idx):
                    valid_filenames.append(filenames[fn][:-11])
                for fi, fn in enumerate(train_idx):
                    train_filenames.append(filenames[fn][:-11])

                self.valid_filenames = valid_filenames
                self.train_filenames = train_filenames

            else:
                print("need validation in config file")
                quit()

    def train_dataloader(self):
        """
        Returns the train dataloader from the train filenames with the specified
        transforms.

        Parameters:None
        Return: DataLoader train_set_loader
        """
        loader_config = self.loader_config
        train_set_loader = DataLoader(
            UniversalDataset(
                self.train_filenames,
                loader_config["PatchPerBuffer"],
                self.size_in,
                self.size_out,
                self.nchannel,
                use_costmap=self.accepts_costmap,
                transforms=self.transforms,
                patchize=True,
                check_crop=self.check_crop,
                init_only=self.init_only,  # first call of train_dataloader is just to get dataset params if init_only is true  # noqa E501
            ),
            batch_size=loader_config["batch_size"],
            shuffle=True,
            num_workers=loader_config["NumWorkers"],
            pin_memory=True,
        )
        return train_set_loader

    def val_dataloader(self):
        """
        Returns the validation dataloader from the validation filenames with
        no transforms

        Parameters: None
        Return: DataLoader val_set_loader
        """
        loader_config = self.loader_config
        val_set_loader = DataLoader(
            UniversalDataset(
                self.valid_filenames,
                loader_config["PatchPerBuffer"],
                self.size_in,
                self.size_out,
                self.nchannel,
                transforms=[],  # no transforms for validation data
                use_costmap=self.accepts_costmap,
                patchize=True,  # validate on entire image
            ),
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=loader_config["NumWorkers"],
            pin_memory=True,
        )
        return val_set_loader

    def test_dataloader(self):
        """
        Returns the test dataloader
        Parameters: None
        Return: DataLoader test_set_loader
        """
        test_set_loader = DataLoader(
            TestDataset(self.config, fns=self.valid_filenames),
            batch_size=1,
            shuffle=False,
            num_workers=self.config["NumWorkers"],
            pin_memory=True,
            worker_init_fn=init_worker,
        )
        return test_set_loader
