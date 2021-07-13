import os
import megengine as mge
from pycocotools import cocoeval

from tools.utils import import_from_file
from hp_module.hyper_param import HyperParam
from hpo_module.evaluator.base_evaluator import EVALUATORS, BaseEvaluator
from omegaconf import OmegaConf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

@EVALUATORS.register
class ChongQiGongMen(BaseEvaluator):
    config_path = HyperParam(name="config_path", default_value="", desc="path to evaluation config file")
    
    def __init__(self, hps):
        super().__init__(hps)
        self.generate_cfg_from_file()
        self.logger = mge.get_logger(__name__)
        self.logger.setLevel("INFO")
    
    def generate_cfg_from_file(self):
        eval_cfg = OmegaConf.load(self.config_path.value)


    def evaluate(self, result_path, cfg):
        eval_gt = COCO(os.path.join("data", cfg.test_dataset["name"], cfg.test_dataset["ann_file"]))
        eval_dt = eval_gt.loadRes(result_path)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType="bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        metrics = [
            "AP",
            "AP@0.5",
            "AP@0.75",
            "APs",
            "APm",
            "APl",
            "AR@1",
            "AR@10",
            "AR@100",
            "ARs",
            "ARm",
            "ARl",
        ]

        performance = {}        
        for i, m in enumerate(metrics):
            performance[m] = cocoEval.stats[i]
        # self.logger.info("mmAP".center(32, "-"))
        # for i, m in enumerate(metrics):
        #     self.logger.info("|\t%s\t|\t%.03f\t|", m, cocoEval.stats[i])
        # self.logger.info("-" * 32)

        return performance