import os
import numpy as np
import time
import datetime
import json
import random
import torch

r"""lys"""

# constant
BITS = 64
EPSILON = 1e-7
MINUS_MAXN = -1e50


class TimeCheck:
    time_last = 0

    @staticmethod
    def present(out=False):
        print(datetime.datetime.now())
        return time.time()

    @staticmethod
    def init(out=False):
        TimeCheck.time_last = time.time()
        if out:
            print("timing start")
        return 0

    @staticmethod
    def check(out=False):
        t = time.time()
        result = t - TimeCheck.time_last
        if out:
            print("time: %f" % (result))
        TimeCheck.time_last = t
        return result


class Config:
    def __init__(self, config_path):
        self.dict = None
        with open(config_path, "r") as f:
            self.dict = json.load(f)
        self.data_path = self.dict["path"]["data"]
        self.dataset_path = self.dict["path"]["data"] + self.dict["path"]["dataset"]
        self.result_path = self.dict["path"]["result"]

        self.gpu = self.dict["gpu"]
        self.seed = self.dict["seed"]
        self.patch_size = (self.dict["patch"]["H"], self.dict["patch"]["W"])

        self.train_workers = self.dict["train"]["workers"]
        self.trainset_rate = self.dict["train"]["trainset_rate"]
        self.epochs = self.dict["train"]["epochs"]
        self.init_lr = self.dict["train"]["init_lr"]
        self.betas = self.dict["train"]["betas"]
        self.loss = self.dict["train"]["loss"]
        self.eval_after_every = self.dict["train"]["eval_after_every"]
        self.batch_size = self.dict["train"]["batch_size"]

        self.eval_workers = self.dict["eval"]["workers"]

    def save(self, config_path):
        with open(config_path, "w") as json_file:
            json.dump(self.dict, json_file)


# os
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_paths(filepath, postfix):
    paths = os.listdir(filepath)
    return [os.path.join(filepath, x) for x in paths if x.endswith(postfix)]


def cut_path(path):
    dir, p = os.path.split(path)
    p_cut = p.split(".")
    p_text = p_cut[0]
    p_postfix = "." + p_cut[-1]
    return dir, p_text, p_postfix


def set_seeds(config: Config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    print("all seeds: %d" % (config.seed))
