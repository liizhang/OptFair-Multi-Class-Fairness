import argparse
from config import *

import os
import yaml
import copy

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', help = 'name of algorithm: erm, postfair, infair',
                        type = str, choices = OPTIMIZERS, default = 'postfair')
    parser.add_argument('--data', help = 'name of dataset:adult, enem, acs, celeba',
                        type = str, default = 'adult')
    # parser.add_argument('--data_setting', help = 'operations on raw datas to generate data set; input a dict',
    #                 type = dict, default = {'sensitive_attr': 'gender', 'generate': True})
    parser.add_argument('--sensitive_attr', help = 'name of sensitive attribute',
                        type = str, default = 'gender')
    parser.add_argument('--generate_data', help = 'whether to generate data',
                        type = bool, default = False)
    # parser.add_argument('--fairness_constraints', help = 'fairness constraints: dp, eo, eopp, range from 0 to 1',
    #                     type = dict, default = {'metric':'dp', 'bound':0.01})
    parser.add_argument('--fairness_metric', help = 'used fairness metric: dp, eo, eopp',
                    type = str, default = 'dp')
    parser.add_argument('--fairness_bound', help = 'used fairness level, range from 0 to infinity',
                    type = float, default = 0.01)
    
    parser.add_argument('--load_param', help = 'used given parameter in file "log/exp_config", if False, use input parameter',
                    type = bool, default = True)

    parser.add_argument('--seed', help = 'operations on raw datas to generate decentralized data set; input a dict',
                        type = int, default = 1)
    # parser.add_argument('--noise', help = 'noise level',
    #                     type = float, default = 0.0)
    
    # model
    parser.add_argument('--model', help = 'name of model;',
                        type = str, choices = MODELS, default = 'logistic')
    parser.add_argument('--loss', help = 'name of model;',
                        type = str, default = 'CE')
    parser.add_argument('--lr', help = 'learning rate for local update',
                        type = float, default = 0.001)
    parser.add_argument('--wd', help = 'weight decay parameter;',
                        type = float, default = 1e-4)
    parser.add_argument('--gpu', help = 'use gpu (default: True)',
                        default = True, action = 'store_true')
    parser.add_argument('--load_model', help = 'if load trained model',
                        type = bool, default = False)

    # parameter
    parser.add_argument('--num_round', help = 'number of rounds to simulate',
                        type = int, default = 50)
    parser.add_argument('--eval_round', help = 'evaluate every ___ rounds',
                        type = int, default = 1)
    parser.add_argument('--batch_size', help = 'batch size when clients train on data',
                        type = int, default = 128)
    parser.add_argument('--optimizer', help = 'optimizer for training',
                        type = str, default = 'AdamW')
    parser.add_argument('--print_result', help = 'if print the result',
                        type = bool, default = True)

    # OptFair general
    parser.add_argument('--dual_params_lr', help = 'personalzied parameter',
                        type = float, default = 0.2)
    parser.add_argument('--dual_params_bound', help = 'personalzied parameter',
                        type = float, default = 10)
    parser.add_argument('--tau', help = 'entropic regularization',
                        type = float, default = 0.02)
    
    ### OptFair-in
    parser.add_argument('--inner_round', help = 'number of rounds to optimize inner problem',
                        type = int, default = 10)
    
    ### OptFair-post
    parser.add_argument('--post_num_round', help = 'number of rounds to simulate',
                        type = int, default = 50)
    parser.add_argument('--post_batch_size', help = 'post batch size when training on data, setting 0 for full batch',
                        type = int, default = 128)

    
    # args = parser.parse_args()

    return parser

def load_yaml(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_update(dst: dict, src: dict) -> dict:
    """merge: src cover dst"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def apply_yaml_defaults(parser: argparse.ArgumentParser, yaml_defaults: dict):
    action_dests = {a.dest for a in parser._actions}

    to_set = {}
    for k, v in (yaml_defaults or {}).items():
        if k not in action_dests:
            continue  # YAML 
        cur = parser.get_default(k)
        if isinstance(cur, dict) and isinstance(v, dict):
            merged = deep_update(copy.deepcopy(cur), v)
            to_set[k] = merged
        else:
            to_set[k] = v

    if to_set:
        parser.set_defaults(**to_set)

def arg_parser():
    parser = build_parser() 

    pre_args, _ = parser.parse_known_args()
    data = getattr(pre_args, "data", None)
    algorithm = getattr(pre_args, "algorithm", None)
    alg_lower = (algorithm or "").lower()
    fairness_metric = getattr(pre_args, "fairness_metric", None)

    if alg_lower == "infair":
        yaml_path = "log/exp_config/in-" + data + '-' + fairness_metric +".yaml"
    elif alg_lower == "postfair":
        yaml_path = "log/exp_config/post-" + data + '-' + fairness_metric + ".yaml"

    if getattr(pre_args, "load_param", True):
        yaml_cfg = load_yaml(yaml_path)
        apply_yaml_defaults(parser, yaml_cfg)

    args = parser.parse_args()
    return args
