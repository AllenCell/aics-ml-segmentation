import pytorch_lightning
from torch.optim import Adam
import torch
from aicsmlsegment.custom_metrics import get_metric
from aicsmlsegment.custom_loss import get_loss_criterion
from aicsmlsegment.model_utils import (
    model_inference,
    apply_on_image,
)
from aicsmlsegment.DataUtils.Universal_Loader import (
    minmax,
    undo_resize,
    UniversalDataset,
    QCDataset,
)
from aicsmlsegment.utils import compute_iou, compute_steps_for_sliding_window

import numpy as np
from skimage.io import imsave
from skimage.morphology import remove_small_objects
import os
import pathlib
from torch.utils.data import DataLoader

# quality control model
class Model(pytorch_lightning.LightningModule):
    # the base class for all the models
    def __init__(self, config, model_config, train):
        super().__init__()

        self.args_inference = {}

        self.model_name = config["model"]["name"]
        self.model_config = model_config

        import importlib
        from aicsmlsegment.model_utils import weights_init as weights_init

        module = importlib.import_module(
            "aicsmlsegment.NetworkArchitecture." + self.model_name
        )
        init_args = {
            "in_channel": model_config["nchannel"],
            "num_classes": model_config["nclass"],
        }

        model = getattr(module, "ResNet3d_18")
        self.model = model(**init_args).apply(weights_init)

        self.args_inference["size_in"] = model_config["size_in"]
        self.args_inference["nclass"] = model_config["nclass"]

        self.config = config
        self.aggregate_img = None
        self.iou_distribution = []
        if train:
            loader_config = config["loader"]
            self.datapath = loader_config["datafolder"]
            self.nworkers = loader_config["NumWorkers"]
            self.batchsize = loader_config["batch_size"]
            self.epoch_shuffle = loader_config["epoch_shuffle"]

            validation_config = config["validation"]
            self.leaveout = validation_config["leaveout"]
            self.validation_period = validation_config["validate_every_n_epoch"]

            self.lr = config["learning_rate"]
            self.weight_decay = config["weight_decay"]

            self.args_inference["inference_batch_size"] = loader_config["batch_size"]
            self.args_inference["OutputCh"] = validation_config["OutputCh"]

            (
                self.loss_function,
                self.accepts_costmap,
            ) = get_loss_criterion(config)
            self.metric = get_metric(config)
            self.scheduler_params = config["scheduler"]
            self.dataset_params = None
        else:
            if config["RuntimeAug"] <= 0:
                self.args_inference["RuntimeAug"] = False
            else:
                self.args_inference["RuntimeAug"] = True
            self.args_inference["OutputCh"] = config["OutputCh"]
            self.args_inference["inference_batch_size"] = config["batch_size"]
            self.args_inference["mode"] = config["mode"]["name"]
            self.args_inference["Threshold"] = config["Threshold"]
            if config["large_image_resize"] != [1, 1, 1]:
                self.aggregate_img = {}
                self.count_map = {}
                self.batch_count = {}
        self.save_hyperparameters()
        self.out_list = []
        self.gt_list = []

    def forward(self, x):
        """
        returns raw predictions
        """
        return self.model(x)

    def configure_optimizers(self):
        optims = []
        scheds = []

        scheduler_params = self.scheduler_params

        # basic optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        optims.append(optimizer)

        if scheduler_params["name"] is not None:
            if scheduler_params["name"] == "ExponentialLR":
                from torch.optim.lr_scheduler import ExponentialLR

                assert scheduler_params["gamma"] > 0
                scheduler = ExponentialLR(
                    optims[0],
                    gamma=scheduler_params["gamma"],
                    verbose=scheduler_params["verbose"],
                )

            elif scheduler_params["name"] == "CosineAnnealingLR":
                from torch.optim.lr_scheduler import CosineAnnealingLR as CALR

                assert scheduler_params["T_max"] > 0
                scheduler = CALR(
                    optims[0],
                    T_max=scheduler_params["T_max"],
                    verbose=scheduler_params["verbose"],
                )

            elif scheduler_params["name"] == "StepLR":
                from torch.optim.lr_scheduler import StepLR

                assert scheduler_params["step_size"] > 0
                assert scheduler_params["gamma"] > 0
                scheduler = StepLR(
                    optims[0],
                    step_size=scheduler_params["step_size"],
                    gamma=scheduler_params["gamma"],
                    verbose=scheduler_params["verbose"],
                )
            elif scheduler_params["name"] == "ReduceLROnPlateau":
                from torch.optim.lr_scheduler import ReduceLROnPlateau

                assert 0 < scheduler_params["factor"] < 1
                assert scheduler_params["patience"] > 0
                # if patience is too short, validation metrics won't be available
                # if "val" in scheduler_params["monitor"]:
                # assert (
                #     scheduler_params["patience"] > self.validation_period
                # ), "Patience must be larger than validation frequency"
                scheduler = ReduceLROnPlateau(
                    optims[0],
                    mode=scheduler_params["mode"],
                    factor=scheduler_params["factor"],
                    patience=scheduler_params["patience"],
                    verbose=scheduler_params["verbose"],
                    min_lr=0.0000001,
                )
                # monitoring metric must be specified
                return {
                    "optimizer": optims[0],
                    "lr_scheduler": scheduler,
                    "monitor": scheduler_params["monitor"],
                }
            elif scheduler_params["name"] == "1cycle":
                from torch.optim.lr_scheduler import OneCycleLR

                scheduler = OneCycleLR(
                    optims[0],
                    max_lr=scheduler_params["max_lr"],
                    total_steps=scheduler_params["total_steps"],
                    pct_start=scheduler_params["pct_start"],
                    verbose=scheduler_params["verbose"],
                )
            else:
                print(
                    "The selected scheduler is not yet supported. No scheduler is used."
                )
                return optims
            scheds.append(scheduler)
            return optims, scheds
        else:
            print("no scheduler is used")
            return optims

    # HACK until pytorch lightning includes reload_dataloaders_every_n_epochs
    def on_train_epoch_start(self):
        if self.epoch_shuffle is not None:
            if self.current_epoch == 0 and self.dataset_params is None:
                self.dataset_params = self.train_dataloader().dataset.get_params()

            if self.current_epoch % self.epoch_shuffle == 0:
                if self.global_rank == 0 and self.current_epoch > 0:
                    print("Reloading dataloader...")
                self.DATALOADER = DataLoader(
                    QCDataset(**self.dataset_params),
                    batch_size=self.config["loader"]["batch_size"],
                    shuffle=True,
                    num_workers=self.config["loader"]["NumWorkers"],
                    pin_memory=True,
                )
            self.iter_dataloader = iter(self.DATALOADER)

    def get_upsample_grid(self, desired_shape, n_targets):
        x = torch.linspace(-1, 1, desired_shape[-1], device=self.device)
        y = torch.linspace(-1, 1, desired_shape[-2], device=self.device)
        z = torch.linspace(-1, 1, desired_shape[-3], device=self.device)
        meshz, meshy, meshx = torch.meshgrid((z, y, x))
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = torch.stack([grid] * n_targets)  # one grid for each target in batch
        return grid

    def log_and_return(self, name, value):
        # sync_dist on_epoch=True ensures that results will be averaged across gpus
        self.log(
            name,
            value,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": value}  # return val only used in train step

    def training_step(self, batch, batch_idx):
        if self.epoch_shuffle is not None:
            # ignore dataloader provided by pytorch lightning
            batch = next(self.iter_dataloader)
            inputs = batch[0].half().to(self.device)
            predictions = batch[1].to(self.device)
            uncertaintymap = batch[2].half().to(self.device)
            label = batch[4].to(self.device)
        else:
            inputs = batch[0]
            predictions = batch[1]
            uncertaintymap = batch[2]
            label = batch[4]

        # print(f'inputs:{inputs.shape}, predictions:{predictions.shape}, uncertaintymap:{uncertaintymap.shape}, label:{label.shape}')
        inputs = torch.cat([inputs, predictions, uncertaintymap], dim=1)
        outputs = self(inputs)
        # print(f'outputs:{outputs.shape}')

        loss = self.loss_function(outputs, label.unsqueeze(dim=1).long())
        return self.log_and_return("epoch_train_loss", loss)

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        predictions = batch[1]
        uncertaintymap = batch[2]
        fn = batch[3]
        label = batch[4]

        inputs = torch.cat([inputs, predictions, uncertaintymap], dim=1)

        with torch.no_grad():
            outputs = self.model.forward(inputs)
        
            # from https://arxiv.org/pdf/1810.11654.pdf, # mode = 'validation'
            val_loss = self.loss_function(outputs, label.unsqueeze(dim=1).long())
            self.log_and_return("val_loss", val_loss)
        # outputs = torch.nn.Softmax(dim=1)(outputs)
        # val_metric = compute_iou(outputs > 0.5, label, torch.unsqueeze(costmap, dim=1))
        # self.log_and_return("val_iou", val_metric)

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        fn = batch["fn"][0] # batch["fn"] is a list
        gt = batch["gt"]
        prediction = batch["prediction"]
        uncertaintymap = batch["uncertaintymap"]
        # print(f'img:{img.shape}')
        print(f'\nfn:{fn}')
        # print(f'gt:{gt.shape}')
        # print(f'prediction:{prediction.shape}')
        # print(f'uncertaintymap:{uncertaintymap.shape}')
        inputs = torch.cat([img.float(), prediction.float(), uncertaintymap.float()], dim=1)
        # patch_size = self.args_inference["size_in"]
        # print(f'patch_size:{patch_size}')
        # steps = compute_steps_for_sliding_window(patch_size, img.shape[2:], 1)
        # print(f'steps:{steps}')
        with torch.no_grad():
            out = self.model.forward(inputs)
            print(f'out:{torch.nn.Softmax(dim=1)(out).cpu().numpy()}, gt:{gt.cpu().numpy()}')
            self.out_list.append(torch.nn.Softmax(dim=1)(out).cpu().numpy())
            self.gt_list.append(gt.cpu().numpy())
        # with torch.no_grad():
        #     for z in steps[0]:
        #         lb_z = z
        #         ub_z = z + patch_size[0]
        #         for y in steps[1]:
        #             lb_y = y
        #             ub_y = y + patch_size[1]
        #             for x in steps[2]:
        #                 lb_x = x
        #                 ub_x = x + patch_size[2]
        #                 # estimated_iou = self.model.forward(inputs[:,:,lb_z:ub_z,lb_y:ub_y,lb_x:ub_x])
        #                 out = self.model.forward(inputs[:,:,lb_z:ub_z,lb_y:ub_y,lb_x:ub_x])
        #                 print(f'estimated_iou:{torch.nn.Softmax(dim=1)(estimated_iou).cpu().numpy()}, gt:{gt.cpu().numpy()}')
        #                 self.out_list.append(torch.nn.Softmax(dim=1)(out).cpu().numpy())
        #                 self.gt_list.append(gt.cpu().numpy())
                        # gt_iou = compute_iou(gt[:,:,lb_z:ub_z,lb_y:ub_y,lb_x:ub_x] > self.args_inference["Threshold"], prediction[:,:,lb_z:ub_z,lb_y:ub_y,lb_x:ub_x] > self.args_inference["Threshold"], None)
                        # if self.model_name == 'resnet3d_18_classification':
                        #     print(f'estimated_iou:{torch.nn.Softmax(dim=1)(estimated_iou).cpu().numpy()}, gt_iou:{gt_iou}')
                        #     estimated_iou_list.append(torch.nn.Softmax(dim=1)(estimated_iou).cpu().numpy()[0])
                        # else:
                        #     print(f'estimated_iou:{estimated_iou.cpu().numpy()}, gt_iou:{gt_iou}')
                        #     estimated_iou_list.append(estimated_iou.cpu().numpy()[0,0])
                        # gt_iou_list.append(gt_iou)
        # original_path = self.config["OutputDir"] + os.sep + pathlib.PurePosixPath(fn).stem
        # path = original_path+"_estimated_iou.npy"
        # np.save(path, np.array(estimated_iou_list))
        # path = original_path+"_gt_iou.npy"
        # np.save(path, np.array(gt_iou_list))
        # print(f'finished')
