import numpy as np
import torch
import os
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import copy
from fvcore.nn import FlopCountAnalysis
from thop import profile

from . import base
from . import data_utils
from . import image_utils
from . import log_utils


LOSS_EPSILON = 1e-3

"""env"""


def env(config: base.Config):
    gpus = ",".join([str(i) for i in config.gpu])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda")
    base.set_seeds(config)
    return device


def check_gpu_mem(device):
    mem = torch.cuda.memory_allocated(device) / 1024 / 1024
    print("now-gpu-mem:%d MB" % mem)


"""dataset"""


# 以下类用于继承
class ImgSet(Dataset):

    def __init__(
        self,
        name,
        src_paths,
        label_paths,
        src_template: image_utils.Image,
        label_template: image_utils.Image,
        patch_size=None,
    ):
        super(ImgSet, self).__init__()
        self.name = name
        self.src_paths = src_paths
        self.label_paths = label_paths
        self.src_template = src_template
        self.label_template = label_template
        self.patch_size = patch_size
        self.size = len(src_paths)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        src_path = self.src_paths[index]
        label_path = self.src_paths[index]
        src = self.src_template.template_load(src_path).to_tensor()
        label = self.label_template.template_load(label_path).to_tensor()
        src, label, self.patch_size, self.offset = data_utils.random_patch(
            src, label, self.patch_size
        )
        return src, label, src_path, label_path

    def batch(self, src, label, src_path, label_path):
        return Batch(src=src, label=label, src_path=src_path, label_path=label_path)


class Batch:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                self.__dict__[k] = v.to(device)


"""framework"""


# 用于继承
class Framework(nn.Module):
    def __init__(self, model_type, loss_type, model_name, config: base.Config):
        super(Framework, self).__init__()
        self.model_type = model_type
        self.loss_type = loss_type
        self.model_name = model_name
        self.config = config
        self.device = env(config)
        self.model = self.get_model()
        self.criterion = self.get_criterion()

    def get_model(self) -> nn.Module:
        pass

    def forward(self, data: Batch):
        return self.model(data.src)

    def get_criterion(self) -> nn.Module:
        type = self.loss_type
        if type == "L1":
            return nn.L1Loss()
        elif type == "Charbonnier":
            return CharbonnierLoss()
        elif type == "RMSE":
            return RMSELoss()
        else:
            return nn.L1Loss()

    def backward(self, model_out, data: Batch):
        loss = self.criterion(model_out, data.label)
        loss.backward()
        return loss

    def eval_model_out(self, model_out: torch.Tensor, data: Batch, metrixes: dict):
        for i in range(model_out.shape[0]):
            result = self.eval_single(model_out[i], data.label[i])
            metrixes[data.src_path[i]] = result
        return metrixes

    def save_model_out(
        self, val_set: ImgSet, model_out: torch.Tensor, data: Batch, out_dir
    ):
        # 适应batch
        for i in range(model_out.shape[0]):
            self.save_single(
                val_set, model_out[i], data.src_path[i], data.label_path[i], out_dir
            )

    @staticmethod
    def eval_single(src: torch.Tensor, label: torch.Tensor):
        result = {}
        result["rmse"] = data_utils.rmse(src, label)
        return result

    @staticmethod
    def save_single(
        val_set: ImgSet, model_out: torch.Tensor, src_path, label_path, out_dir
    ):
        val_set.label_template.from_tensor(model_out)
        _, src_name, _ = base.cut_path(src_path)
        _, _, label_postfix = base.cut_path(label_path)
        val_set.label_template.filename = os.path.join(
            out_dir, src_name + label_postfix
        )
        val_set.label_template.save()
        pass

    """running"""

    # training，暂不支持多卡parallel
    def train(
        self,
        config: base.Config,
        train_set: ImgSet,
        val_set: ImgSet,
        resume=True,  # 继续训练的checkpoint名
        save_model_out=False,
    ):
        # prerparation
        # if not resume:
        #     self.model_name = self.model_type
        resume_name = self.model_name
        self.model_name += "_" + train_set.name + "_" + str(round(time.time()))
        model_name = self.model_name

        model_path = os.path.join(config.result_path, "checkpoints", model_name)
        base.mkdir(model_path)

        # logger
        log_path = os.path.join(model_path, "log.log")
        log_utils.logger_info(model_name, log_path)
        logger = log_utils.logging.getLogger(model_name)
        logger.info("train-%s" % (model_name))

        # optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.init_lr, betas=config.betas
        )

        # scheduler
        start_epoch = 0
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epochs - warmup_epochs, eta_min=1e-6
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=scheduler_cosine,
        )

        # resume & model & loss
        self.to(self.device)
        if resume:
            logger.info("resume training")
            resume_path = os.path.join(config.result_path, "checkpoints", resume_name)
            self.model, optimizer, scheduler, start_epoch = load_checkpoint(
                resume_path, self.model, optimizer, scheduler, latest=True
            )
        else:
            logger.info("new training")
        # self.to(self.device)

        # train dataset
        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.train_workers,
            drop_last=True,
        )

        # run
        start_epoch += 1
        max_epoch = config.epochs + 1
        logger.info("===> Start Epoch {} End Epoch {}".format(start_epoch, max_epoch))
        logger.info("===> Loading datasets")
        logger.info("train loader size: %d" % (len(train_loader)))
        print("===> Loading datasets")
        loss_list = []

        for epoch in range(start_epoch, max_epoch):
            base.TimeCheck.init()
            epoch_loss = 0
            self.model = self.model.train()
            opt_lr = scheduler.get_last_lr()
            logger.info(
                "##==========={}-training, Epoch: {}, lr: {} =============##".format(
                    "fp32", epoch, opt_lr
                )
            )
            # train
            for i, args in enumerate(tqdm(train_loader), 0):
                # zero_grad
                optimizer.zero_grad()
                data = train_set.batch(*args)
                data.to(self.device)
                loss = self.backward(self.forward(data), data)
                optimizer.step()
                epoch_loss += loss.item()
            loss_list.append(epoch_loss)

            # evaluate
            if epoch != 0 and epoch % config.eval_after_every == 0:
                metrix, details = self.evaluate(
                    config,
                    val_set,
                    save_model_out,
                    on_device=True,
                )
                logger.info(metrix)
                logger.info(details)
                checkpoint(
                    model_path, self.model, optimizer, scheduler, epoch, logger, True
                )

            # end
            t = base.TimeCheck.check()
            scheduler.step()
            logger.info(
                "===> Epoch {} Complete: Lr {:.6f} Avg. Loss: {:.6f} time: {:.2f}".format(
                    epoch, scheduler.get_lr()[0], epoch_loss / config.batch_size, (t)
                )
            )
        data_utils.show_line(
            np.arange(epoch),
            np.array(loss_list),
            title="loss",
            vis=False,
            save_path=os.path.join(model_path, "loss.png"),
        )
        return loss_list

    # test
    def test(
        self,
        config: base.Config,
        test_set: ImgSet,
        save_model_out=False,
    ):
        # prerparation
        model_name = self.model_name
        model_path = os.path.join(config.result_path, "checkpoints", model_name)
        log_path = os.path.join(model_path, "log.log")

        # logger
        log_utils.logger_info(model_name, log_path)
        logger = log_utils.logging.getLogger(model_name)
        logger.info("test-%s on %s" % (model_name, test_set.name))

        # resume
        logger.info("start_testing")
        self.model, _, _, start_epoch = load_checkpoint(
            model_path, self.model, latest=True
        )

        # model
        self.model.to(self.device)
        # check_gpu_mem(self.device)

        # evaluate
        metrix, details = self.evaluate(
            config,
            test_set,
            save_model_out,
            on_device=True,
        )
        logger.info(metrix)
        logger.info(details)
        # end
        return metrix

    def evaluate(
        self,
        config: base.Config,
        val_set: ImgSet,
        save_model_out=False,
        on_device=False,  # 模型是不是已经在device上
    ):
        with torch.no_grad():
            metrixes = {}
            # model
            if not on_device:
                self.model.to(self.device)
            model_name = self.model_name

            # val_set
            val_loader = DataLoader(
                val_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.eval_workers,
                drop_last=False,
            )

            self.model = self.model.eval()
            model_out_dir = os.path.join(
                config.result_path, "output", val_set.name, model_name
            )
            base.mkdir(model_out_dir)

            for i, args in enumerate(tqdm(val_loader), 0):
                data = val_set.batch(*args)
                # check_gpu_mem(self.device)
                data.to(self.device)
                model_out = self.forward(data)
                self.eval_model_out(model_out, data, metrixes)
                model_out = model_out.cpu()
                if save_model_out:
                    self.save_model_out(val_set, model_out, data, model_out_dir)

            metrix = {}
            for k in metrixes.keys():
                for key in metrixes[k].keys():
                    # 去张量化
                    metrixes[k][key] = float(metrixes[k][key])
                    if key not in metrix.keys():
                        metrix[key] = []
                    metrix[key].append(metrixes[k][key])

            for key, value in metrix.items():
                metrix[key] = np.mean(np.array(value), axis=0)
        return metrix, metrixes

    def complexity(self, config: base.Config, inputs):
        self.model.to(self.device)
        # prerparation
        model_name = self.model_name
        model_path = os.path.join(config.result_path, "checkpoints", model_name)
        log_path = os.path.join(model_path, "log.log")

        # logger
        log_utils.logger_info(model_name, log_path)
        logger = log_utils.logging.getLogger(model_name)

        flops, n_param = profile(self.model, inputs)
        # flops = FlopCountAnalysis(self.model, inputs)
        # n_param = sum([p.nelement() for p in self.model.parameters()])
        # logger.info("FLOPs: {:.2f} G".format(flops.total() / (1000 * 1000 * 1000)))
        logger.info("FLOPs: {:.2f} G".format(flops / (1000 * 1000 * 1000)))
        logger.info("Params: {:.2f} M".format(n_param / 1000 / 1000))
        return flops, n_param


def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def is_frozen(model: nn.Module):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)


def checkpoint(dir, model, optimizer, scheduler, epoch, logger, latest=False):
    with torch.no_grad():
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        if latest:
            model_out_path = os.path.join(dir, "model_latest.pth")
        else:
            model_out_path = os.path.join(dir, "model_epoch_{}.pth".format(epoch))
        torch.save(state_dict, model_out_path)
        logger.info("Checkpoint saved to {}".format(model_out_path))


def load_checkpoint(dir, model, optimizer=None, scheduler=None, epoch=0, latest=True):
    with torch.no_grad():
        if latest:
            model_in_path = os.path.join(dir, "model_latest.pth")
        else:
            model_in_path = os.path.join(dir, "model_epoch_{}.pth".format(epoch))
        checkpoint = torch.load(model_in_path)
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        return model, optimizer, scheduler, epoch


"""losses"""


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (LOSS_EPSILON * LOSS_EPSILON)))
        return loss


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        return data_utils.rmse(x, y)
