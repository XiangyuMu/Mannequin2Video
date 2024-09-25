import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

# from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import socket
from pytorch_lightning.plugins.environments import ClusterEnvironment, SLURMEnvironment

# tensorboard
from pytorch_lightning.loggers import TensorBoardLogger

# rich
from rich import traceback


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # 训练的名字，与resume不可同时定义
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    # 继续训练的checkpoints的路径
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["configs/stable-diffusion/v1-inference-inpaint.yaml"],
    )
    # 默认名字的后缀
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    # 训练结果保存路径
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    # 设置随机数种子
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    # 是否从头开始训练
    parser.add_argument(
        "--train_from_scratch",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Train from scratch",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="path to pretrained model",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )


#找出opt对象中与Trainer默认中不同的参数，并且返回这些参数名的排序列表
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    # 实例化参数解析器
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)  # 将参数输入到Trainer中

    opt, unknown = parser.parse_known_args()   # 相当于在控制台中如果输入了未知的参数，不会报错，而是会存在unknow中
    
    # name与resume不能同时被定义，其中resume是指定从某个checkpoint开始训练
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    # 如果resume已被定义，则读取resume的checkpoint
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))  # 将logdir下的configs文件夹下的所有yaml文件按照字母顺序排序
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    # 如果resume未被定义，则根据name来定义logdir
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    
    ckptdir = os.path.join(logdir, "checkpoints")  # 保存checkpoint的路径
    cfgdir = os.path.join(logdir, "configs") # 保存配置文件的路径
    seed_everything(opt.seed) # 设置随机数种子

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]  # 从base中读取配置文件
    cli = OmegaConf.from_dotlist(unknown)  # 将unknow中的参数转换为OmegaConf格式
    config = OmegaConf.merge(*configs, cli)  # 合并配置文件
    lightning_config = config.pop("lightning", OmegaConf.create()) # 从配置文件中读取lightning配置
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) # 从lightning配置中读取trainer配置

    trainer_config["accelerator"] = "ddp"  # 将加速器设置为ddp模式
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)  # 将opt中的非trainer默认参数添加到trainer_config中
    if not "gpus" in trainer_config:  # 如果trainer_config中没有gpus参数，则设置为0
        del trainer_config["accelerator"]
        cpu = True
    else:  # 如果trainer_config中有gpus参数，则设置为gpu
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config  # 将trainer_config添加到lightning_config中

    # 实例化模型（总模型）
    model = instantiate_from_config(config.model)
    if not opt.resume:
        if opt.train_from_scratch:   # 如果train_from_scratch为True，则从头开始训练
            ckpt_file = torch.load(opt.pretrained_model, map_location="cpu")[
                "state_dict"
            ]  # 加载预训练模型到cpu上
            ckpt_file = {
                key: value
                for key, value in ckpt_file.items()
                if not (key[:6] == "model.")
            } # 从预训练模型中去除model.前缀
            model.load_state_dict(ckpt_file, strict=False)  # 加载预训练模型，并且不完全严格匹配
            print("Train from scratch!")
        else:
            model.load_state_dict(
                    torch.load(opt.pretrained_model, map_location="cpu")["state_dict"],
                    strict=False,
                )
            
    # trainer and callbacks
    trainer_kwargs = dict()

    # tensorboard logger
    logger = TensorBoardLogger(save_dir=logdir, name="my_model")  # tensorboard日志保存路径logdir/my_model
    trainer_kwargs["logger"] = logger
#-------------------------------------------------------------------------------------
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",  # 指定目标回调类型
        "params": {
            "dirpath": ckptdir,  # 指定检查点文件的保存路径
            "filename": "{epoch:06}",  # 指定检查点文件的文件名（例如epoch000001）
            "verbose": True,  # 表示在保存检查点输出的详细信息
            "save_last": False,  # 表示是否保存最后一个检查点
        },
    }  # 配置PyTorch Lightning的模型检查点回调
    if hasattr(model, "monitor"):  # 如果模型有monitor属性
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor # 设置检查点回调的监控指标
        default_modelckpt_cfg["params"]["save_top_k"] = 5  # 保存前5个检查点

    if "modelcheckpoint" in lightning_config:   # 如果lightning_config中有modelcheckpoint配置
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg) # 合并默认的模型检查点配置和用户自定义的模型检查点配置
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):  # 如果pytorch-lightning版本小于1.4.0，则设置trainer_kwargs中的checkpoint_callback为modelckpt_cfg
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)
#-------------------------------------------------------------------------------------

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {  # 设置回调的名称
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {  # 配置一个名为image_logger的回调
            "target": "main.ImageLogger",
            "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
        },
        "learning_rate_logger": {   # 配置一个名为learning_rate_logger的回调
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
        "cuda_callback": {"target": "main.CUDACallback"},  # 配置一个名为cuda_callback的回调
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})
# 提取lightning_config中的callbacks配置
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:  # 如果callbacks_cfg中有metrics_over_trainsteps_checkpoint配置
        print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)


    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)  # 合并默认的回调配置和用户自定义的回调配置
    if "ignore_keys_callback" in callbacks_cfg and hasattr(  
        trainer_opt, "resume_from_checkpoint"  # 如果trainer_opt中有resume_from_checkpoint属性
    ):
        callbacks_cfg.ignore_keys_callback.params[
            "ckpt_path"
        ] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]
# 创建一个callback的列表，别表中的每个元素都是根据callbacks_cfg的实例化的回调对象
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs, precision=16)  # 根据trainer_opt和trainer_kwargs创建一个Trainer对象

    trainer.logdir = logdir  # 设置trainer的logdir属性



    data = instantiate_from_config(config.data)  # 实例化数据集
    data.prepare_data()
    data.setup() # 设置数据集
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}") # 打印数据集的信息

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate # 从配置文件中读取batch_size和base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer: # 梯度积累
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches  # 如果trainer中有accumulate_grad_batches参数，则设置为该参数的值
    else:
        accumulate_grad_batches = 1

    num_nodes = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches  # 设置trainer的accumulate_grad_batches属性
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr  # 设置学习率
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_nodes) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate,
                accumulate_grad_batches,
                num_nodes,
                ngpu,
                bs,
                base_lr,
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
#---------------------------------------------------------------------
    if opt.train:  
        try:
            trainer.fit(model, data)  # 训练模型·
        except Exception:
            # melk()  # don't save last checkpoint
            raise
    if not opt.no_test and not trainer.interrupted:  # 如果不是no_test并且没有被中断，则测试模型
        trainer.test(model, data)



class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[
            worker_id * split_size : (worker_id + 1) * split_size
        ]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)





class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        # self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.num_workers = 20
        # self.num_workers = 0

        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(
            self.datasets["train"], Txt2ImgIterableBaseDataset
        )
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else True,
            worker_init_fn=init_fn,
        )

    def _val_dataloader(self, shuffle=False):
        if (
            isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset)
            or self.use_worker_init_fn
        ):
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(
            self.datasets["train"], Txt2ImgIterableBaseDataset
        )
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _predict_dataloader(self, shuffle=False):
        if (
            isinstance(self.datasets["predict"], Txt2ImgIterableBaseDataset)
            or self.use_worker_init_fn
        ):
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
        )
