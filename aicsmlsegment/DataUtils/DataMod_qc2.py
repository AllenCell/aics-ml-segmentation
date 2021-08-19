from aicsmlsegment.DataUtils.Universal_Loader import (
    UniversalDataset,
    TestDataset,
    QCDataset,
    QCTestDataset,
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


class DataModule_qc(pytorch_lightning.LightningDataModule):
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

        if train:
            self.loader_config = config["loader"]

            name = self.loader_config["name"]
            if name not in ["default", "focus"]:
                print("other loaders are under construction")
                quit()

            self.check_crop = False
            self.transforms = []
            if "Transforms" in self.loader_config:
                self.transforms = self.loader_config["Transforms"]

            _, self.accepts_costmap = get_loss_criterion(config)

            self.init_only = False
            if self.loader_config["epoch_shuffle"] is not None:
                self.init_only = True

            model_config = config["model"]
            self.size_in = model_config["size_in"]
            self.nchannel = 1   # set to 1 because the image is one channel

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
        if stage == "fit":  # no setup is required for testing
            # load settings #
            config = self.config

            # get validation and training filenames from input dir from config
            validation_config = config["validation"]
            loader_config = config["loader"]
            if validation_config["metric"] is not None:

                if type(loader_config["datafolder"]) == str:
                    loader_config["datafolder"] = [loader_config["datafolder"]]

                filenames = []
                pseudo_labels = []
                for folder in loader_config["datafolder"]:
                    fns = glob(folder + "/positive/*_img.tiff")
                    fns.sort()
                    filenames += fns
                    pseudo_labels += [1 for _ in fns]
                    fns = glob(folder + "/negative/*_img.tiff")
                    fns.sort()
                    filenames += fns
                    pseudo_labels += [0 for _ in fns]

                total_num = len(filenames)
                LeaveOut = validation_config["leaveout"]
                if len(LeaveOut) == 1:
                    if LeaveOut[0] > 0 and LeaveOut[0] < 1:
                        num_train = int(np.floor((1 - LeaveOut[0]) * total_num))
                        shuffled_idx = np.arange(total_num)
                        # make sure validation sets are same across gpus
                        random.seed(0)
                        random.shuffle(shuffled_idx)
                        train_idx = shuffled_idx[:num_train]
                        valid_idx = shuffled_idx[num_train:]
                        rand = True
                    else:
                        valid_idx = [int(LeaveOut[0])]
                        train_idx = list(
                            set(range(total_num)) - set(map(int, LeaveOut))
                        )
                valid_filenames = []
                train_filenames = []
                valid_labels = []
                train_labels = []
                # remove file extensions from filenames
                for fi, fn in enumerate(valid_idx):
                    valid_filenames.append(filenames[fn][:-9])
                    valid_labels.append(pseudo_labels[fn])
                for fi, fn in enumerate(train_idx):
                    train_filenames.append(filenames[fn][:-9])
                    train_labels.append(pseudo_labels[fn])

                self.valid_filenames = valid_filenames
                self.train_filenames = train_filenames
                self.valid_labels = valid_labels
                self.train_labels = train_labels
                print(f'train_filenames:{self.train_filenames[:10]}')
                print(f'train_labels:{self.train_labels[:10]}')
                print(f'valid_filenames:{self.valid_filenames[:10]}')
                print(f'valid_labels:{self.valid_labels[:10]}')

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
            QCDataset(
                self.train_filenames,
                self.train_labels,
                loader_config["PatchPerBuffer"],
                self.size_in,
                self.nchannel,
                use_uncertaintymap=True,
                transforms=self.transforms,
                patchize=False,
                init_only=self.init_only,  # first call of train_dataloader is just to get dataset params if init_only is true  # noqa E501
                uncertainty_type=loader_config["uncertainty_type"],
                threshold=loader_config["threshold"],
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
            QCDataset(
                self.valid_filenames,
                self.valid_labels,
                loader_config["PatchPerBuffer"],
                self.size_in,
                self.nchannel,
                use_uncertaintymap=True,
                transforms=[],
                patchize=False,
                uncertainty_type=loader_config["uncertainty_type"],
                threshold=loader_config["threshold"],
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
        """
        # maybe you want to do the test on your validation set, let's get all the files of the validation set
        filenames = []
        config = self.config
        valid_filenames = None
        if config["mode"]["name"] == 'cv':
            folder = config["mode"]["InputDir"]
            fns = glob(folder + "/*_GT.ome.tif")
            fns.sort()
            filenames += fns
            total_num = len(filenames)
            LeaveOut = config["validation"]["leaveout"]
            if len(LeaveOut) == 1:
                if LeaveOut[0] > 0 and LeaveOut[0] < 1:
                    num_train = int(np.floor((1 - LeaveOut[0]) * total_num))
                    shuffled_idx = np.arange(total_num)
                    # make sure validation sets are same across gpus
                    random.seed(0)
                    random.shuffle(shuffled_idx)
                    train_idx = shuffled_idx[:num_train]
                    valid_idx = shuffled_idx[num_train:]
                else:
                    valid_idx = [int(LeaveOut[0])]
                    train_idx = list(
                        set(range(total_num)) - set(map(int, LeaveOut))
                    )
            valid_filenames = []
            train_filenames = []
            # remove file extensions from filenames
            for fi, fn in enumerate(valid_idx):
                valid_filenames.append(filenames[fn][:-11])
            for fi, fn in enumerate(train_idx):
                train_filenames.append(filenames[fn][:-11])
            # print(f'valid_filenames:{valid_filenames}')
        test_set_loader = DataLoader(
            QCTestDataset(self.config, fns=valid_filenames),
            batch_size=1,
            shuffle=False,
            num_workers=self.config["NumWorkers"],
            pin_memory=True,
            worker_init_fn=init_worker,
        )
        return test_set_loader
        """
        filenames = []
        pseudo_labels = []
        folder = self.config["mode"]["InputDir"]
        fns = glob(folder + "/positive/*_img.tiff")
        fns.sort()
        filenames += fns
        pseudo_labels += [1 for _ in fns]
        fns = glob(folder + "/negative/*_img.tiff")
        fns.sort()
        filenames += fns
        pseudo_labels += [0 for _ in fns]
        test_set_loader = DataLoader(
            QCTestDataset(self.config, fns=filenames, pseudo_labels=pseudo_labels),
            batch_size=1,
            shuffle=False,
            num_workers=self.config["NumWorkers"],
            pin_memory=True,
            worker_init_fn=init_worker,
        )
        return test_set_loader