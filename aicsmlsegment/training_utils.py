# import torch.nn.functional as F
# from torch import nn as nn
# from torch.autograd import Variable
import logging

import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import random
from glob import glob
from tqdm import tqdm

from aicsimageio import imread

from aicsmlsegment.custom_loss import MultiAuxillaryElementNLLLoss
from aicsmlsegment.model_utils import save_checkpoint, model_inference
from aicsmlsegment.utils import (
    compute_iou,
    get_logger,
)


def shuffle_split_filenames(datafolder, leaveout):
    print("prepare the data ... ...")
    filenames = glob(datafolder + "/*_GT.ome.tif")
    filenames.sort()
    total_num = len(filenames)
    if len(leaveout) == 1:
        if leaveout[0] > 0 and leaveout[0] < 1:
            num_train = int(np.floor((1 - leaveout[0]) * total_num))
            shuffled_idx = np.arange(total_num)
            random.shuffle(shuffled_idx)
            train_idx = shuffled_idx[:num_train]
            valid_idx = shuffled_idx[num_train:]
        else:
            valid_idx = [int(leaveout[0])]
            train_idx = list(set(range(total_num)) - set(map(int, leaveout)))
    elif leaveout:
        valid_idx = list(map(int, leaveout))
        train_idx = list(set(range(total_num)) - set(valid_idx))

    valid_filenames = []
    train_filenames = []
    for _, fn in enumerate(valid_idx):
        valid_filenames.append(filenames[fn][:-11])
    for _, fn in enumerate(train_idx):
        train_filenames.append(filenames[fn][:-11])

    return train_filenames, valid_filenames


def _log_lr(self):
    lr = self.optimizer.param_groups[0]["lr"]
    self.writer.add_scalar("learning_rate", lr, self.num_iterations)


def _log_stats(self, phase, loss_avg, eval_score_avg):
    tag_value = {
        f"{phase}_loss_avg": loss_avg,
        f"{phase}_eval_score_avg": eval_score_avg,
    }

    for tag, value in tag_value.items():
        self.writer.add_scalar(tag, value, self.num_iterations)


def _log_params(self):
    self.logger.info("Logging model parameters and gradients")
    for name, value in self.model.named_parameters():
        self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
        self.writer.add_histogram(
            name + "/grad", value.grad.data.cpu().numpy(), self.num_iterations
        )


def _log_images(self, input, target, prediction):
    sources = {
        "inputs": input.data.cpu().numpy(),
        "targets": target.data.cpu().numpy(),
        "predictions": prediction.data.cpu().numpy(),
    }
    for name, batch in sources.items():
        for tag, image in self._images_from_batch(name, batch):
            self.writer.add_image(tag, image, self.num_iterations, dataformats="HW")
