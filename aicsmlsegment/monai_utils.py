from glob import glob
import pytorch_lightning
from monai.networks.layers import Norm
from monai.networks.nets import BasicUNet

from torch import nn, squeeze

from aicsmlsegment.model_utils import (
    model_inference,
)
import random

from aicsmlsegment.custom_loss import DiceLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

SUPPORTED_MONAI_MODELS = [
    "BasicUNet",
]
SUPPORTED_LOSSES = ["Aux"]


class Monai_BasicUNet(pytorch_lightning.LightningModule):
    def __init__(self, config, train):
        super().__init__()
        self._model = BasicUNet(
            dimensions=len(config["size_in"]),
            in_channels=config["nchannel"],
            out_channels=config["nclass"],
            norm=Norm.BATCH,
        )
        self.args_inference = lambda: None

        if train:
            self.loss_function = DiceLoss()  # config["loss"]["name"]
            self.datapath = config["loader"]["datafolder"]
            self.leaveout = config["validation"]["leaveout"]
            self.nworkers = config["loader"]["NumWorkers"]
            self.batchsize = config["loader"]["batch_size"]
            self.lr = config["learning_rate"]
            self.weight_decay = config["weight_decay"]

            self.args_inference.OutputCh = config["validation"]["OutputCh"]
        else:
            self.args_inference.OutputCh = config["OutputCh"]

        self.args_inference.size_in = config["size_in"]
        self.args_inference.size_out = config["size_out"]
        self.args_inference.nclass = config["nclass"]

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        return Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch[0].cuda()
        targets = batch[1]
        outputs = self.forward(inputs)
        # select output channel
        outputs = outputs[:, self.args_inference.OutputCh, :, :, :]

        if len(targets) > 1:
            for zidx in range(len(targets)):
                targets[zidx] = targets[zidx].cuda()
        else:
            targets = targets[0].cuda()
        if len(batch) == 3:  # input + target + cmap
            cmap = batch[2].cuda()
            loss = self.loss_function(outputs, targets)  # , cmap)
        else:  # input + target
            loss = self.loss_function(outputs, targets)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_img = batch[0].cuda()
        label = batch[1].cuda()
        costmap = batch[2].cuda()

        if len(input_img.shape) == 3:
            # add channel dimension
            input_img = np.expand_dims(input_img, axis=0)
        elif len(input_img.shape) == 4:
            # assume number of channel < number of Z, make sure channel dim comes first
            if input_img.shape[0] > input_img.shape[1]:
                input_img = np.transpose(input_img, (1, 0, 2, 3))

        outputs = model_inference(
            self._model, input_img[0].cpu(), nn.Softmax(dim=1), self.args_inference
        )
        val_loss = self.loss_function(outputs, squeeze(label, dim=1))
        return {"val_loss": val_loss}


class DataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.loader_config = config["loader"]

    def prepare_data(self):
        pass

    def setup(self, stage):
        ### load settings ###
        config = self.config  # TODO, fix this

        # dataloader
        validation_config = config["validation"]
        loader_config = config["loader"]
        if validation_config["metric"] is not None:
            print("prepare the data ... ...")
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
                    train_idx = list(set(range(total_num)) - set(map(int, LeaveOut)))
            elif LeaveOut:
                valid_idx = list(map(int, LeaveOut))
                train_idx = list(set(range(total_num)) - set(valid_idx))

            valid_filenames = []
            train_filenames = []
            for fi, fn in enumerate(valid_idx):
                valid_filenames.append(filenames[fn][:-11])
            for fi, fn in enumerate(train_idx):
                train_filenames.append(filenames[fn][:-11])

            self.valid_filenames = valid_filenames
            self.train_filenames = train_filenames

        else:
            # TODO, update here
            print("need validation")
            quit()

    def train_dataloader(self):
        loader_config = self.config["loader"]
        config = self.config
        if loader_config["name"] == "default":
            from aicsmlsegment.DataLoader3D.Universal_Loader import (
                RR_FH_M0 as train_loader,
            )

            train_set_loader = DataLoader(
                train_loader(
                    self.train_filenames,
                    loader_config["PatchPerBuffer"],
                    config["size_in"],
                    config["size_out"],
                ),
                num_workers=loader_config["NumWorkers"],
                batch_size=loader_config["batch_size"],
                shuffle=True,
            )
        elif loader_config["name"] == "focus":
            from aicsmlsegment.DataLoader3D.Universal_Loader import (
                RR_FH_M0C as train_loader,
            )

            train_set_loader = DataLoader(
                train_loader(
                    self.train_filenames,
                    loader_config["PatchPerBuffer"],
                    config["size_in"],
                    config["size_out"],
                ),
                num_workers=loader_config["NumWorkers"],
                batch_size=loader_config["batch_size"],
                shuffle=True,
            )
        else:
            print("other loader not support yet")
            quit()
        return train_set_loader

    def val_dataloader(self):
        from aicsmlsegment.DataLoader3D.Universal_Loader import NOAUG_M as val_loader

        val_set_loader = DataLoader(
            val_loader(
                self.valid_filenames,
                self.loader_config["PatchPerBuffer"],
                self.config["size_in"],
                self.config["size_out"],
            ),
            num_workers=self.loader_config["NumWorkers"],
            batch_size=1,
            shuffle=False,
        )
        return val_set_loader
