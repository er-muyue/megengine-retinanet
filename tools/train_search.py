import argparse
import os, sys, copy
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf

from loguru import logger
from hpo import register_module
from hpo_module.config import hpo_default_cfg
from hpo_module.engine.builder import build_tester, build_trainer
from hpo_module.evaluator.builder import build as build_evaluator
from hpo_module.search_manager import SearchManager
from hpo_module.search_policy.builder import build as build_search_policy
from hpo_module.utils.utils import get_exp_name_with_param, save_param_performance
from tools.utils import AverageMeter, DetectionPadCollator, GroupedRandomSampler, parse_config, get_config_info, import_from_file


class NewSearchManager(SearchManager):
    def __init__(self, base_config, search_param_config, hpo_config, search_policy, evaluator, analyzer, trainer, testesr):
        super().__init__(base_config, search_param_config, hpo_config, search_policy, evaluator, analyzer, trainer, testesr)

    def update_hp(self, target_cfg, hpo_schedules):
        for i in range(len(hpo_schedules)):
            setattr(target_cfg, hpo_schedules[i][0], hpo_schedules[i][1])
        return target_cfg

    def run_single_pipeline(self, epoch_iter, search_iter, search_param):
        logger.info(f"epoch {epoch_iter+1}/{self.hpo_config.max_search_epoch}")
        logger.info(f"iter {search_iter+1}")
        param_performance = dict()
        train_test_config = copy.deepcopy(self.base_config.Cfg())
        self.update_hp(train_test_config, search_param)
        self.trainer.update_exp_name_with_param(get_exp_name_with_param(search_param))
        self.tester.update_exp_name_with_param(get_exp_name_with_param(search_param))
        self.trainer.train(train_test_config)
        # self.tester.test(train_test_config)
        result_path = self.tester.get_result_path()
        param_performance["param"] = search_param
        param_performance["model_path"] = self.trainer.get_model_path()
        # param_performance["performance"] = self.evaluator.evaluate(result_path, train_test_config)

        # save_param_performance(self.param_performance_file, param_performance)



def main():
    parser = argparse.ArgumentParser(description="Megengine-retinanet search training")
    # parser.add_argument("--log_file", "-log", type=str, default="train_log.txt")
    parser.add_argument("-cfg", default="", metavar="FILE", type=str, help="path to base config file")
    parser.add_argument("-hpo", default="", metavar="FILE", type=str, help="path to hpo config file")
    # parser.add_argument("--continue", "-c", type=str, metavar="FILE", dest="continue_fpath", help="continue from one certain checkpoint")
    # parser.add_argument("--output_dir", "-o", default="data", help="output directory")
    # parser.add_argument("--resume", "-re", action="store_true", help="if resume the searched result")
    parser.add_argument("-f", "--file", default="net.py", type=str, help="net description file")
    parser.add_argument("-w", "--weight_file", default=None, type=str, help="weights file")
    # parser.add_argument("--performance_file", "-pf", default="", help="the path to searched performance file")
    parser.add_argument("--devices", default=1, type=int, help="total number of gpus for testing",)
    parser.add_argument("-n", "--ngpus", default=2, type=int, help="total number of gpus for training")
    parser.add_argument("-b", "--batch_size", default=2, type=int, help="batchsize for training")
    parser.add_argument("-d", "--dataset_dir", default="data", type=str)
    parser.add_argument("-se", "--start_epoch", default=-1, type=int)
    parser.add_argument("-ee", "--end_epoch", default=-1, type=int)
    args = parser.parse_args()

    parse_cfg_file = import_from_file(args.cfg)
    # cfg2dic = parse_config(base_cfg)
    hpo = OmegaConf.load(args.hpo)
    hpo_cfg = OmegaConf.merge(hpo_default_cfg, hpo.hpo)
    evaluator = build_evaluator(hpo_cfg.evaluator)
    hpo_param_cfg = hpo.param
    sp = build_search_policy(hpo_cfg.search_policy, OmegaConf.to_container(hpo_param_cfg))
    trainer = build_trainer(hpo_cfg.trainer, parser=parser)
    tester = build_tester(hpo_cfg.tester, parser=parser)

    search_manager = NewSearchManager(parse_cfg_file, hpo_param_cfg, hpo_cfg, sp, evaluator, None, trainer, tester)
    search_manager.run()
    

if __name__ == "__main__":
    main()
