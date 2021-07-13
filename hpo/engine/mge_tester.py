import os
import json
from multiprocessing import Process, Queue
from tqdm import tqdm
import megengine as mge
import megengine.distributed as dist
from megengine.data import DataLoader
from hp_module.hyper_param import HyperParam
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.utils import import_from_file
from hpo_module.engine.engine_base import TESTERS, TesterBase
from tools.data_mapper import data_mapper
from tools.utils import DetEvaluator, InferenceSampler, import_from_file
from models.retinanet import RetinaNet

@TESTERS.register
class MGETester(TesterBase):
    config_path = HyperParam(name="config_path", default_value="", desc="path to config file")
    
    def __init__(self, hps, parser):
        super().__init__(hps)
        self.args = parser.parse_args()
        self.cur_result_path = None
        self.logger = mge.get_logger(__name__)
        self.logger.setLevel("INFO")

    def get_result_path(self):
        return self.cur_result_path

    def update_exp_name_with_param(self, param_name):
        self.param_name = param_name

    def test(self, config):
        
        current_network = import_from_file(self.args.cfg)

        if self.args.weight_file:
            self.args.start_epoch = self.args.end_epoch = -1
        else:
            if self.args.start_epoch == -1:
                self.args.start_epoch = config.max_epoch - 1
            if self.args.end_epoch == -1:
                self.args.end_epoch = self.args.start_epoch
            assert 0 <= self.args.start_epoch <= self.args.end_epoch < config.max_epoch

        for epoch_num in range(self.args.start_epoch, self.args.end_epoch + 1):
            if self.args.weight_file:
                weight_file = self.args.weight_file
            else:
                weight_file = "search-{}/{}/epoch_{}.pkl".format(
                    os.path.basename(self.args.cfg).split(".")[0], self.param_name, epoch_num
                )

            result_list = []
            # if self.args.devices > 1:
            #     result_queue = Queue(2000)

            #     master_ip = "localhost"
            #     server = dist.Server()
            #     port = server.py_server_port
            #     procs = []
            #     for i in range(self.args.devices):
            #         proc = Process(
            #             target=self.worker,
            #             args=(
            #                 current_network,
            #                 config,
            #                 weight_file,
            #                 self.args.dataset_dir,
            #                 result_queue,
            #                 master_ip,
            #                 port,
            #                 self.args.devices,
            #                 i,
            #             ),
            #         )
            #         proc.start()
            #         procs.append(proc)

            #     num_imgs = dict(coco=5000, objects365=30000, chongqigongmen=97)

            #     for _ in tqdm(range(num_imgs[config.test_dataset["name"]])):
            #         result_list.append(result_queue.get())

            #     for p in procs:
            #         p.join()
            # else:
            self.worker(current_network, config, weight_file, self.args.dataset_dir, result_list)

            all_results = DetEvaluator.format(result_list, config)
            json_path = "search-{}/{}/epoch_{}.json".format(os.path.basename(self.args.cfg).split(".")[0], self.param_name, epoch_num)
            all_results = json.dumps(all_results)

            with open(json_path, "w") as fo:
                fo.write(all_results)
            self.cur_result_path = json_path
            self.logger.info("Save to %s finished, start evaluation!", json_path)

    def worker(self, current_network, cfg, weight_file, dataset_dir, result_list, master_ip=None, port=None, world_size=1, rank=0):
        if world_size > 1:
            dist.init_process_group(
                master_ip=master_ip,
                port=port,
                world_size=world_size,
                rank=rank,
                device=rank,
            )

        # setattr(cfg, "backbone_pretrained", False)
        
        # model = current_network.Net(cfg)
        model = RetinaNet(cfg)
        model.eval()

        state_dict = mge.load(weight_file)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

        evaluator = DetEvaluator(model)

        test_loader = self.build_dataloader(dataset_dir, cfg)
        if dist.get_world_size() == 1:
            test_loader = tqdm(test_loader)

        for data in test_loader:
            image, im_info = DetEvaluator.process_inputs(
                data[0][0],
                cfg.test_image_short_size,
                cfg.test_image_max_size,
            )
            pred_res = evaluator.predict(
                image=mge.tensor(image),
                im_info=mge.tensor(im_info)
            )
            # print(pred_res)
            result = {
                "det_res": pred_res,
                "image_id": int(data[1][3][0]),  # "image_id": int(data[1][2][0].split(".")[0].split("_")[-1]),
            }
            if dist.get_world_size() > 1:
                result_list.put_nowait(result)
            else:
                result_list.append(result)

    def build_dataloader(self, dataset_dir, cfg):
        val_dataset = data_mapper[cfg.test_dataset["name"]](
            os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["root"]),
            os.path.join(dataset_dir, cfg.test_dataset["name"], cfg.test_dataset["ann_file"]),
            order=["image", "info"],
        )
        val_sampler = InferenceSampler(val_dataset, 1)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
        return val_dataloader
