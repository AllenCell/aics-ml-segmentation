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
)
from aicsmlsegment.utils import compute_iou

import numpy as np
from skimage.io import imsave
from skimage.morphology import remove_small_objects
import os
import pathlib
from torch.utils.data import DataLoader


class Model(pytorch_lightning.LightningModule):
    # the base class for all the models
    def __init__(self, config, model_config, train):
        super().__init__()

        self.args_inference = {}

        self.model_name = config["model"]["name"]
        self.model_config = model_config

        if "unet_xy" in self.model_name:  # custom model
            import importlib
            from aicsmlsegment.model_utils import weights_init as weights_init

            module = importlib.import_module(
                "aicsmlsegment.NetworkArchitecture." + self.model_name
            )
            init_args = {
                "in_channel": model_config["nchannel"],
                "n_classes": model_config["nclass"],
                "test_mode": not train,
            }
            if self.model_name == "sdunet_xy":
                init_args["loss"] = config["loss"]["name"]

            if "zoom" in self.model_name:
                init_args["down_ratio"] = model_config.get("zoom_ratio", 3)

            model = getattr(module, "UNet3D")
            self.model = model(**init_args).apply(weights_init)

            self.args_inference["size_in"] = model_config["size_in"]
            self.args_inference["size_out"] = model_config["size_out"]
            self.args_inference["nclass"] = model_config["nclass"]

        else:  # monai model
            if self.model_name == "segresnetvae":
                from monai.networks.nets.segresnet import SegResNetVAE as model

                model_config["input_image_size"] = model_config["patch_size"]
            elif self.model_name == "extended_vnet":
                from aicsmlsegment.NetworkArchitecture.vnet import VNet as model
            elif self.model_name == "extended_dynunet":
                from aicsmlsegment.NetworkArchitecture.dynunet import DynUNet as model

            else:
                import importlib

                module = importlib.import_module(
                    "monai.networks.nets." + self.model_name
                )
                # deal with monai name scheme - module name != class name for networks
                net_name = [attr for attr in dir(module) if "Net" in attr][0]
                model = getattr(module, net_name)
            # monai model assumes same size for input and output
            self.args_inference["size_in"] = model_config["patch_size"]
            self.args_inference["size_out"] = model_config["patch_size"]
            del model_config["patch_size"]
            self.model = model(**model_config)

        self.config = config
        self.aggregate_img = None
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
                    UniversalDataset(**self.dataset_params),
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
            targets = batch[1].to(self.device)
            cmap = batch[2].to(self.device)
        else:
            inputs = batch[0]
            targets = batch[1]
            cmap = batch[2]
        outputs = self(inputs)

        vae_loss = 0
        if self.model_name == "segresnetvae":
            # segresnetvae forward returns an additional vae loss term
            outputs, vae_loss = outputs
        if (
            "dynunet" in self.model_name and self.model_config["deep_supervision"]
        ):  # output is a stacked tensor all of same shape instead of a list
            outputs = torch.unbind(outputs, dim=1)

        # from https://arxiv.org/pdf/1810.11654.pdf,
        # vae_loss > 0 if model = segresnetvae
        loss = self.loss_function(outputs, targets) + 0.1 * vae_loss
        return self.log_and_return("epoch_train_loss", loss)

    def validation_step(self, batch, batch_idx):
        input_img = batch[0]
        label = batch[1]
        costmap = batch[2]
        fn = batch[3]

        outputs, vae_loss = model_inference(
            self.model,
            input_img,
            self.args_inference,
            squeeze=True,
            extract_output_ch=False,
            model_name=self.model_name,
            softmax=False,
        )
        # from https://arxiv.org/pdf/1810.11654.pdf, # mode = 'validation'
        val_loss = self.loss_function(outputs, label) + 0.1 * vae_loss
        self.log_and_return("val_loss", val_loss)
        outputs = torch.nn.Softmax(dim=1)(outputs)
        val_metric = compute_iou(outputs > 0.7, label, None)
        self.log_and_return("val_iou", val_metric)
        """
        # save all validation result in the given epoch, for quality control network training
        if self.current_epoch == 79 or self.current_epoch == 99:
            # compute uncertainty
            with torch.no_grad():
                output_img_list = []
                for i in range(10):
                    outputs, vae_loss = model_inference(
                        self.model,
                        input_img,
                        self.args_inference,
                        squeeze=True,
                        extract_output_ch=False,
                        model_name=self.model_name,
                        softmax=False,
                    )
                    outputs = torch.nn.Softmax(dim=1)(outputs)
                    output_img_list.append(outputs.detach().cpu().numpy())
            # dimension of the output_img_list would be [mc,N,C,Z,Y,X]
            output_img_list = np.array(output_img_list)
            output_img = output_img_list.mean(axis=0)
            uncertaintymap_variance = np.mean(np.square(output_img_list), 0) - np.square(np.mean(output_img_list, 0))[:,-1,...]
            entropy = -np.sum(np.mean(output_img_list, 0) * np.log(np.mean(output_img_list, 0) + 1e-5), 1) # entropy sum on the C dimension
            expected_entropy = -np.mean(np.sum(output_img_list * np.log(output_img_list + 1e-5), 2), 0)
            uncertaintymap_entropy = entropy
            uncertaintymap_mi = entropy - expected_entropy
            for i in range(input_img.shape[0]):
                # save original image
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+".tiff", input_img[i,0,:,:,:].detach().cpu().numpy())
                # save the prediction
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_prediction.tiff", output_img[i, 1, :, :, :])
                # save cmap
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_cmap.tiff", costmap[i, :, :, :].detach().cpu().numpy())
                # save the ground truth
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_gt.tiff", label[i, :, :, :].detach().cpu().numpy())
                # save uncertainty
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_uncertainty_entropy.tiff", uncertaintymap_variance[i, :, :, :])
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_uncertainty_variance.tiff", uncertaintymap_entropy[i, :, :, :])
                imsave(self.config["checkpoint_dir"]+os.sep+"validation_results"+os.sep+pathlib.PurePosixPath(fn[i]).stem+"_uncertainty_mi.tiff", uncertaintymap_mi[i, :, :, :])
        """
        # save first validation image result
        if batch_idx == 0:
            imsave(
                self.config["checkpoint_dir"]
                + os.sep
                + "validation_results"
                + os.sep
                + "epoch="
                + str(self.current_epoch)
                + "_loss="
                + str(round(val_loss.item(), 3))
                + "_iou="
                + str(round(val_metric, 3))
                + ".tiff",
                outputs[0, 1, :, :, :].detach().cpu().numpy(),
            )
            imsave(
                self.config["checkpoint_dir"]
                + os.sep
                + "validation_results"
                + os.sep
                + "epoch="
                + str(self.current_epoch)
                + "_loss="
                + str(round(val_loss.item(), 3))
                + "_iou="
                + str(round(val_metric, 3))
                + "_GT.tiff",
                label[0, 0, :, :, :].detach().cpu().numpy(),
            )
        

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        fn = batch["fn"][0]
        tt = batch["tt"][0]
        save_n_batches = batch["save_n_batches"].detach().cpu().numpy()[0]
        args_inference = self.args_inference
        to_numpy = True
        if self.aggregate_img is not None:
            to_numpy = False  # prevent excess gpu->cpu data transfer

        # enable dropout layer during inference
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                m.eval()

        mc_num = 16
        output_img_list = []
        uncertaintymap_predvariance_list = []
        print(f'now predicting file:{fn}')
        for i in range(mc_num):
            print(f'now predicting the {i}/{mc_num} mc sample...')
            output_img, _ = apply_on_image(
                self.model,
                img,
                args_inference,
                squeeze=True,
                to_numpy=False,
                softmax=False,
                model_name=self.model_name,
                extract_output_ch=False,
            )
            # if the model name is unet_xy_zoom_0pad_predvar, the last feature map of dimension 1 should be the predicted uncertainty map
            if self.model_name == 'unet_xy_zoom_0pad_predvar':
                output_img_list.append(torch.nn.Softmax(dim=1)(output_img[:,:-1,...]).detach().cpu().numpy())
                uncertaintymap_predvariance_list.append(output_img[:,-1,...].detach().cpu().numpy())
            else:
                output_img_list.append(torch.nn.Softmax(dim=1)(output_img).detach().cpu().numpy())

        print('now calculating uncertainty...')
        # dimension of the output_img_list would be [10,1,C,Z,Y,X]
        output_img_list = np.array(output_img_list)
        output_img = output_img_list.mean(axis=0)[:,-1,...]
        # just use the first maximum softmax as teh uncertaintymap_softmax
        uncertaintymap_softmax = np.max(output_img_list[0], 1)
        # entropy code adapted from https://github.com/tanyanair/segmentation_uncertainty
        uncertaintymap_variance = (np.mean(np.square(output_img_list), 0) - np.square(np.mean(output_img_list, 0))).squeeze(0)[-1,...]
        # output_img_list = np.repeat(output_img_list, 2, 2) # expand the class dimension
        # output_img_list[:,:,0,...] = 1 - output_img_list[:,:,1,...]
        entropy = -np.sum(np.mean(output_img_list, 0) * np.log(np.mean(output_img_list, 0) + 1e-5), 1)
        expected_entropy = -np.mean(np.sum(output_img_list * np.log(output_img_list + 1e-5), 2), 0)
        uncertaintymap_entropy = entropy.squeeze(0)
        uncertaintymap_mi = (entropy - expected_entropy).squeeze(0)
        uncertaintymap_predvariance = None
        if len(uncertaintymap_predvariance_list):
            uncertaintymap_predvariance = np.mean(np.array(uncertaintymap_predvariance_list),0).squeeze(0)

        # Here, we compute the mutual iou of each mc sample and save the result in a numpy
        # mutual_iou_list = []
        # volume_list = [np.count_nonzero(output_img_list[i]>0.5) for i in range(output_img_list.shape[0])]
        # for i in range(output_img_list.shape[0]-1):
        #     for j in range(i,output_img_list.shape[0]):
        #         current_iou = compute_iou(output_img_list[i] > 0.5, output_img_list[j] > 0.5, None)
        #         mutual_iou_list.append(current_iou)

        if self.aggregate_img is not None:
            # initialize the aggregate img
            i, j, k = batch["ijk"][0], batch["ijk"][1], batch["ijk"][2]
            if fn not in self.aggregate_img:
                self.aggregate_img[fn] = torch.zeros(
                    batch["im_shape"], dtype=torch.float32, device=self.device
                )
                self.count_map[fn] = torch.zeros(
                    batch["im_shape"], dtype=torch.uint8, device=self.device
                )
                self.batch_count[fn] = 0
            self.aggregate_img[fn][
                :,  # preserve all channels
                i : i + output_img.shape[2],
                j : j + output_img.shape[3],
                k : k + output_img.shape[4],
            ] += torch.squeeze(output_img, dim=0)

            self.count_map[fn][
                :,  # preserve all channels
                i : i + output_img.shape[2],
                j : j + output_img.shape[3],
                k : k + output_img.shape[4],
            ] += 1
            self.batch_count[fn] += 1
        else:
            # not aggregating an image, save every batch
            self.batch_count = {fn: 1}

        # only want to perform post-processing and saving once the aggregated image
        # is complete or we're not aggregating an image
        if self.batch_count[fn] % save_n_batches == 0:
            from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

            if self.aggregate_img is not None:
                # normalize for overlapping patches
                output_img = self.aggregate_img[fn] / self.count_map[fn]
                output_img = output_img.detach().cpu().numpy()

            if args_inference["mode"] != "folder":
                out = minmax(output_img)
                out = undo_resize(out, self.config)
                if args_inference["Threshold"] > 0:
                    out = out > args_inference["Threshold"]
                    out = out.astype(np.uint8)
                    out[out > 0] = 255
            else:
                if args_inference["Threshold"] < 0:
                    out = minmax(output_img)
                    out = undo_resize(out, self.config)
                    out = minmax(out)
                else:
                    out = remove_small_objects(
                        output_img > args_inference["Threshold"],
                        min_size=2,
                        connectivity=1,
                    )
                    out = out.astype(np.uint8)
                    out[out > 0] = 255
            out = np.squeeze(out, 0)  # remove N dimension
            original_path = self.config["OutputDir"] + os.sep + pathlib.PurePosixPath(fn).stem
            if tt != -1:
                path = path + "_T_" + f"{tt:03}"

            print('now saving predictions...')
            path = original_path+"_struct_segmentation.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=out,
                    channel_names=[self.config["segmentation_name"]],
                    dimension_order="CZYX",
                )
            # also save the softmax segmentation
            path = original_path+"_softmax_segmentation.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=np.squeeze(output_img, 0),
                    channel_names=[self.config["segmentation_name"]],
                    dimension_order="CZYX",
                )
            # save uncertainty map
            if uncertaintymap_softmax is not None: uncertaintymap_softmax = 1 - np.squeeze(uncertaintymap_softmax, 0)
            path = original_path+"_uncertaintymap_softmax.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=uncertaintymap_softmax,
                    channel_names=['uncertainty'],
                    dimension_order="CZYX",
                )
            path = original_path+"_uncertaintymap_entropy.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=uncertaintymap_entropy,
                    channel_names=['uncertainty'],
                    dimension_order="CZYX",
                )
            path = original_path+"_uncertaintymap_variance.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=uncertaintymap_variance,
                    channel_names=['uncertainty'],
                    dimension_order="CZYX",
                )
            path = original_path+"_uncertaintymap_mi.tiff"
            with OmeTiffWriter(path, overwrite_file=True) as writer:
                writer.save(
                    data=uncertaintymap_mi,
                    channel_names=['uncertainty'],
                    dimension_order="CZYX",
                )
            if uncertaintymap_predvariance is not None:
                path = original_path+"_uncertaintymap_predvariance.tiff"
                with OmeTiffWriter(path, overwrite_file=True) as writer:
                    writer.save(
                        data=uncertaintymap_predvariance,
                        channel_names=['uncertainty'],
                        dimension_order="CZYX",
                    )
            # # save iou agreement in mc samples
            # path = original_path+"_iou_mc.npy"
            # np.save(path,np.array(mutual_iou_list))
            # # save mc volume variance
            # path = original_path+"_volume_mc.npy"
            # np.save(path, volume_list)
