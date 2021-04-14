from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from typing import Dict

SUPPORTED_LOSSES = {
    # MONAI
    "Dice": {
        "source": "monai.losses",
        "args": ["softmax", "include_background"],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": False},
    },
    "GeneralizedDice": {
        "source": "monai.losses",
        "args": ["softmax", "include_background"],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": False},
    },
    "Focal": {
        "source": "monai.losses",
        "args": [],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": False},
    },
    "MaskedDice": {
        "source": "monai.losses",
        "args": ["softmax", "include_background"],
        "wrapper_args": {
            "n_label_ch": 2,
            "accepts_costmap": True,
            "cmap_unsqueeze": True,
        },
    },
    # TORCH
    "MSE": {
        "source": "torch.nn",
        "args": [],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": False},
    },
    "CrossEntropy": {
        "source": "torch.nn",
        "args": [],
        "wrapper_args": {
            "n_label_ch": 1,
            "accepts_costmap": False,
            "to_long": True,
            "label_squeeze": True,
        },
    },
    # CUSTOM
    "PixelWiseCrossEntropy": {
        "source": "aicsmlsegment.custom_loss",
        "args": [],
        "costmap": True,
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": True},
    },
    "ElementAngularMSE": {
        "source": "aicsmlsegment.custom_loss",
        "args": [],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": True},
    },
    "MaskedMSE": {
        "source": "aicsmlsegment.custom_loss",
        "args": [],
        "wrapper_args": {"n_label_ch": 2, "accepts_costmap": True},
    },
    "MaskedCrossEntropy": {
        "source": "aicsmlsegment.custom_loss",
        "args": [],
        "wrapper_args": {
            "n_label_ch": 1,
            "accepts_costmap": True,
            "cmap_unsqueeze": True,
            "label_squeeze": True,
            "to_long": True,
        },
    },
    "MultiAuxillaryCrossEntropy": {
        "source": "aicsmlsegment.custom_loss",
        "args": ["weight", "num_class"],
        "wrapper_args": {
            "n_label_ch": 1,
            "accepts_costmap": True,
            "cmap_unsqueeze": True,
            "label_squeeze": True,
            "to_long": True,
        },
    },
}


def get_loss_criterion(config: Dict):
    """
    Returns the loss function based on provided configuration

    Parameters
    ----------
    config: Dict
        a top level configuration object containing the 'loss' key

    Return:
    -------------
    an instance of the loss function and whether it accepts a costmap and loss weights
    """
    import importlib

    name = config["loss"]["name"]
    # backwards compatibility
    if name == "Aux":
        name = "MultiAuxillaryCrossEntropy"

    loss_names = [name]
    if "+" in name:
        loss_names = name.split("+")
    losses = []
    costmap = []
    for ln in loss_names:
        assert (
            ln in SUPPORTED_LOSSES
        ), f'Invalid loss: {ln}. Supported losses: {[key for key in SUPPORTED_LOSSES]} or combinations as "l1+l2"'
        loss_info = SUPPORTED_LOSSES[ln]

        init_args = loss_info["args"]

        module = importlib.import_module(loss_info["source"])
        module = getattr(module, ln + "Loss")
        args = {}
        if "softmax" in init_args:
            args["softmax"] = True
        if "num_task" in init_args:
            args["num_task"] = len(config["model"]["nclass"])
        if "weight" in init_args:
            args["weight"] = config["loss"]["loss_weight"]
        if "num_class" in init_args:
            args["num_class"] = config["model"]["nclass"]
        if "include_background" in init_args:
            args["include_background"] = False
        loss = module(**args)
        wrapped_loss = LossWrapper(loss, **loss_info["wrapper_args"])
        losses.append(wrapped_loss)
        costmap.append(loss_info["wrapper_args"]["accepts_costmap"])

    if len(losses) == 2:
        from aicsmlsegment.custom_loss import CombinedLoss

        return CombinedLoss(*losses), np.any(costmap)

    else:
        return losses[0], np.any(costmap)


class LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss,
        n_label_ch,
        accepts_costmap,
        to_long=False,
        cmap_unsqueeze=False,
        label_squeeze=False,
    ):
        super(LossWrapper, self).__init__()

        self.loss = loss
        self.n_label_ch = n_label_ch
        self.cmap_unsqueeze = cmap_unsqueeze
        self.label_squeeze = label_squeeze
        self.accepts_costmap = accepts_costmap
        self.to_long = to_long

    def forward(self, input, target, cmap=None):
        print("PRE:", input.shape, target.shape, end=" ")
        if cmap is not None:
            print(cmap.shape)
        if self.n_label_ch == 2:
            target = torch.squeeze(torch.stack([1 - target, target], dim=1), dim=2)
        if self.cmap_unsqueeze:
            cmap = torch.unsqueeze(cmap, dim=1)
        if self.to_long:
            target = target.long()
        if self.label_squeeze:
            target = torch.squeeze(target, dim=1)
        # THIS HAPPENS ON LAST OUTPUT OF extended_dynunet, not sure why
        if type(input) == tuple:
            input = input[0]
        print("POST:", input.shape, target.shape, end=" ")
        if self.accepts_costmap and cmap is not None:
            print(cmap.shape)
            loss = self.loss(input, target, cmap)
        else:
            loss = self.loss(input, target)
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss1, loss2):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2

    def forward(self, input, target, cmap=None):
        if cmap is not None:
            loss1_result = self.loss1(input, target, cmap)
            loss2_result = self.loss2(input, target, cmap)
        else:
            loss1_result = self.loss1(input, target)
            loss2_result = self.loss2(input, target)
        return loss1_result + loss2_result


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss(reduction="none")
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target, cmap):
        """
        expects input, target, cmap in NCZYX with input channels = 2, target_channels = 1
        """
        loss = self.loss(self.log_softmax(input), target)
        loss = torch.mean(torch.mul(loss.view(loss.numel()), cmap.view(cmap.numel())))
        return loss


class MultiAuxillaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight, num_class):
        super(MultiAuxillaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.loss_fn = MaskedCrossEntropyLoss()

    def forward(self, input, target, cmap):
        if not isinstance(input, list):  # custom model validation
            input = [input]
        total_loss = self.weight[0] * self.loss_fn(input[0], target, cmap)
        for n in np.arange(1, len(input)):
            total_loss += self.weight[n] * self.loss_fn(input[n], target, cmap)

        return total_loss


class ElementNLLLoss(torch.nn.Module):
    def __init__(self, num_class):
        super(ElementNLLLoss, self).__init__()
        self.num_class = num_class

    def forward(self, input, target, weight):
        target_np = target.detach().cpu().data.numpy()
        target_np = target_np.astype(np.uint8)

        row_num = target_np.shape[0]
        mask = np.zeros((row_num, self.num_class))
        mask[np.arange(row_num), target_np] = 1

        class_x = torch.masked_select(
            input, Variable(torch.from_numpy(mask).cuda().bool())
        )

        out = torch.mul(class_x, weight)
        loss = torch.mean(torch.neg(out), 0)

        return loss


class MultiAuxillaryElementNLLLoss(torch.nn.Module):
    def __init__(self, num_task, weight, num_class):
        super(MultiAuxillaryElementNLLLoss, self).__init__()
        self.num_task = num_task
        self.weight = weight

        self.criteria_list = []
        for n in range(self.num_task):
            self.criteria_list.append(ElementNLLLoss(num_class[n]))

    def forward(self, input, target, cmap):

        total_loss = self.weight[0] * self.criteria_list[0](
            input[0], target.view(target.numel()), cmap.view(cmap.numel())
        )

        for n in np.arange(1, self.num_task):
            total_loss = total_loss + self.weight[n] * self.criteria_list[n](
                input[n], target.view(target.numel()), cmap.view(cmap.numel())
            )

        return total_loss


class MultiTaskElementNLLLoss(torch.nn.Module):
    def __init__(self, weight, num_class):
        super(MultiTaskElementNLLLoss, self).__init__()
        self.num_task = len(num_class)
        self.weight = weight

        self.criteria_list = []
        for n in range(self.num_task):
            self.criteria_list.append(ElementNLLLoss(num_class[n]))

    def forward(self, input, target, cmap):

        assert len(target) == self.num_task and len(input) == self.num_task

        total_loss = self.weight[0] * self.criteria_list[0](
            input[0], target[0].view(target[0].numel()), cmap.view(cmap.numel())
        )

        for n in np.arange(1, self.num_task):
            total_loss = total_loss + self.weight[n] * self.criteria_list[n](
                input[n], target[n].view(target[n].numel()), cmap.view(cmap.numel())
            )

        return total_loss


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, weight):
        return (
            torch.sum(torch.mul((input - target) ** 2, weight))
            / torch.gt(weight, 0).data.nelement()
        )


class ElementAngularMSELoss(torch.nn.Module):
    def __init__(self):
        super(ElementAngularMSELoss, self).__init__()

    def forward(self, input, target, weight):

        # ((input - target) ** 2).sum() / input.data.nelement()
        return (
            torch.sum(
                torch.mul(
                    torch.acos(torch.sum(torch.mul(input, target), dim=1)) ** 2, weight
                )
            )
            / torch.gt(weight, 0).data.nelement()
        )


def compute_per_channel_dice(
    input, target, epsilon=1e-5, ignore_index=None, weight=None
):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2.0 * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(
        self,
        epsilon=1e-5,
        weight=None,
        ignore_index=None,
        sigmoid_normalization=True,
        skip_last_target=False,
    ):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(
            input,
            target,
            epsilon=self.epsilon,
            ignore_index=self.ignore_index,
            weight=weight,
        )
        # Average the Dice score across all channels/classes
        return torch.mean(1.0 - per_channel_dice)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf"""

    def __init__(
        self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True
    ):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(
            1.0 / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False
        )

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect

        denominator = (input + target).sum(-1) * class_weights

        return torch.mean(1.0 - 2.0 * intersect / denominator.clamp(min=self.epsilon))


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf"""

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(
            input, target, weight=class_weights, ignore_index=self.ignore_index
        )

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class BCELossWrapper:
    """
    Wrapper around BCE loss functions allowing to pass 'ignore_index' as well as 'skip_last_target' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, "ignore_index"):
            raise RuntimeError(
                f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead"
            )
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]

        assert input.size() == target.size()

        masked_input = input
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            masked_input = input * mask
            masked_target = target * mask

        return self.loss_criterion(masked_input, masked_target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)

        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(
            target[:, 0, :, :, :], C=input.size()[1], ignore_index=self.ignore_index
        )
        # expand weights
        # weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(
                target.data.ne(self.ignore_index).float(), requires_grad=False
            )
            log_probabilities = log_probabilities * mask
            target = target * mask

        # apply class weights
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights
        class_weights = class_weights.view(1, input.size()[1], 1, 1, 1)
        class_weights = Variable(class_weights, requires_grad=False)
        # add class_weights to each channel
        weights = class_weights + weights
        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(0)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        src = src.type(torch.int64)
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)
