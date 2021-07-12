import argparse
import bisect
import copy
from models.retinanet import RetinaNet
import multiprocessing as mp
import os
import time
import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.optimizer import SGD

from mge_tools.data_mapper import data_mapper
from mge_tools.utils import AverageMeter, DetectionPadCollator, GroupedRandomSampler, get_config_info, import_from_file

from hpo_module.engine.engine_base import TRAINERS, TrainerBase

logger = mge.get_logger(__name__)
logger.setLevel("INFO")
mge.device.set_prealloc_config(1024, 1024, 512 * 1024 * 1024, 2.0)


@TRAINERS.register
class MGETrainer(TrainerBase):
    def __init__(self, hps, parser):
        super().__init__(hps)
        self.parser = parser
        self.cur_model_path = None
        self.args = self.parser.parse_args()

    def get_model_path(self):
        return self.cur_model_path

    def train(self, config):
        # ------------------------ begin training -------------------------- #
        logger.info("Device Count = %d", self.args.ngpus)
        
        # log_dir = "search-{}".format(os.path.basename(self.args.cfg).split(".")[0])
        # os.makedirs(log_dir, exist_ok=True)

        if self.args.ngpus > 1:
            master_ip = "localhost"
            port = dist.get_free_ports(1)[0]
            dist.Server(port)
            processes = list()
            for rank in range(self.args.ngpus):
                process = mp.Process(
                    target=self.worker, args=(master_ip, port, self.args.ngpus, rank, config)
                )
                process.start()
                processes.append(process)

            for p in processes:
                p.join()
        else:
            self.worker(None, None, 1, 0, config)

    def worker(self, master_ip, port, world_size, rank, config):
        if world_size > 1:
            dist.init_process_group(
                master_ip=master_ip,
                port=port,
                world_size=world_size,
                rank=rank,
                device=rank,
            )
            logger.info("Init process group for gpu{} done".format(rank))
        current_network = import_from_file(self.args.cfg)
        model = current_network.Net(config)
        model.train()

        # if dist.get_rank() == 0:
        #     logger.info(get_config_info(config))
            # logger.info(repr(model))

        params_with_grad = []
        for name, param in model.named_parameters():
            if "bottom_up.conv1" in name and config.backbone_freeze_at >= 1:
                continue
            if "bottom_up.layer1" in name and config.backbone_freeze_at >= 2:
                continue
            params_with_grad.append(param)

        opt = SGD(
            params_with_grad,
            lr=config.basic_lr * self.args.batch_size,
            momentum=config.momentum,
            weight_decay=config.weight_decay * dist.get_world_size(),
        )

        gm = GradManager()
        if dist.get_world_size() > 1:
            gm.attach(
                params_with_grad,
                callbacks=[dist.make_allreduce_cb("SUM", dist.WORLD)]
            )
        else:
            gm.attach(params_with_grad)

        if self.args.weight_file is not None:
            weights = mge.load(self.args.weight_file)
            model.backbone.bottom_up.load_state_dict(weights, strict=False)
        if dist.get_world_size() > 1:
            dist.bcast_list_(model.parameters(), dist.WORLD)  # sync parameters

        if dist.get_rank() == 0:
            logger.info("Prepare dataset")
        train_loader = iter(self.build_dataloader(self.args.batch_size, self.args.dataset_dir, config))

        for epoch in range(config.max_epoch):
            self.train_one_epoch(model, train_loader, opt, gm, epoch, config, self.args)
            if dist.get_rank() == 0:
                save_dir = "search-{}/{}".format(os.path.basename(self.args.cfg).split(".")[0], self.param_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = "search-{}/{}/epoch_{}.pkl".format(
                    os.path.basename(self.args.cfg).split(".")[0], self.param_name, epoch
                )
                self.cur_model_path = save_path
                mge.save(
                    {"epoch": epoch, "state_dict": model.state_dict()}, save_path,
                )
                logger.info("dump weights to %s", save_path)

    def train_one_epoch(self, model, data_queue, opt, gm, epoch, cfg, args):
        def train_func(image, im_info, gt_boxes):
            with gm:
                loss_dict = model(image=image, im_info=im_info, gt_boxes=gt_boxes)
                gm.backward(loss_dict["total_loss"])
                loss_list = list(loss_dict.values())
            opt.step().clear_grad()
            return loss_list

        meter = AverageMeter(record_len=cfg.num_losses)
        time_meter = AverageMeter(record_len=2)
        log_interval = cfg.log_interval
        tot_step = cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())
        for step in range(tot_step):
            self.adjust_learning_rate(opt, epoch, step, cfg, args)

            data_tik = time.time()
            mini_batch = next(data_queue)
            data_tok = time.time()

            tik = time.time()
            loss_list = train_func(
                image=mge.tensor(mini_batch["data"]),
                im_info=mge.tensor(mini_batch["im_info"]),
                gt_boxes=mge.tensor(mini_batch["gt_boxes"])
            )
            tok = time.time()

            time_meter.update([tok - tik, data_tok - data_tik])

            if dist.get_rank() == 0:
                info_str = "e%d, %d/%d, lr:%f, "
                loss_str = ", ".join(
                    ["{}:%f".format(loss) for loss in cfg.losses_keys]
                )
                time_str = ", train_time:%.3fs, data_time:%.3fs"
                log_info_str = info_str + loss_str + time_str
                meter.update([loss.numpy() for loss in loss_list])
                if step % log_interval == 0:
                    logger.info(
                        log_info_str,
                        epoch,
                        step,
                        tot_step,
                        opt.param_groups[0]["lr"],
                        *meter.average(),
                        *time_meter.average()
                    )
                    meter.reset()
                    time_meter.reset()

    def adjust_learning_rate(self, optimizer, epoch, step, cfg, args):
        base_lr = (
            cfg.basic_lr * args.batch_size * (
                cfg.lr_decay_rate
                ** bisect.bisect_right(cfg.lr_decay_stages, epoch)
            )
        )
        # Warm up
        lr_factor = 1.0
        if epoch == 0 and step < cfg.warm_iters:
            lr_factor = (step + 1.0) / cfg.warm_iters
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_factor

    def update_exp_name_with_param(self, param_name):
        self.param_name = param_name

    def build_dataset(self, dataset_dir, cfg):
        data_cfg = copy.deepcopy(cfg.train_dataset)
        data_name = data_cfg.pop("name")

        data_cfg["root"] = os.path.join(dataset_dir, data_name, data_cfg["root"])

        if "ann_file" in data_cfg:
            data_cfg["ann_file"] = os.path.join(dataset_dir, data_name, data_cfg["ann_file"])

        data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]

        return data_mapper[data_name](**data_cfg)


    # pylint: disable=dangerous-default-value
    def build_sampler(self, train_dataset, batch_size, aspect_grouping=[1]):
        def _compute_aspect_ratios(dataset):
            aspect_ratios = []
            for i in range(len(dataset)):
                info = dataset.get_img_info(i)
                aspect_ratios.append(info["height"] / info["width"])
            return aspect_ratios

        def _quantize(x, bins):
            return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

        if len(aspect_grouping) == 0:
            return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

        aspect_ratios = _compute_aspect_ratios(train_dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


    def build_dataloader(self, batch_size, dataset_dir, cfg):
        train_dataset = self.build_dataset(dataset_dir, cfg)
        train_sampler = self.build_sampler(train_dataset, batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            transform=T.Compose(
                transforms=[
                    T.ShortestEdgeResize(
                        cfg.train_image_short_size,
                        cfg.train_image_max_size,
                        sample_style="choice",
                    ),
                    T.RandomHorizontalFlip(),
                    T.ToMode(),
                ],
                order=["image", "boxes", "boxes_category"],
            ),
            collator=DetectionPadCollator(),
            num_workers=2,
        )
        return train_dataloader
