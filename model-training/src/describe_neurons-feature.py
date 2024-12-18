import os
import argparse
import datetime
import json
import pandas as pd
import torch
import numpy as np
import utils_old as utils
import data_utils
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import configparser



def save_features(clip_name, target_name, target_layers, d_probe, 
                     batch_size, device, pool_mode, save_dir, weights=None):
    
    target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_t = data_utils.get_data(d_probe, target_preprocess)
    save_names = utils.get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = '',
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    if target_layers==['last_layer']:
        utils.get_last_layer_feature(data_t,target_save_name,batch_size=batch_size,device=device,weights=weights, target_model=target_model, target_preprocess=target_preprocess, target_name=target_name)
        print("1")
    else: 
        utils.save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode,weights)
        print("2")

    return

config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"]
root_path = ROOT
current_path = root_path + '/model-training'

# print("current_path:", current_path)

# raise Exception(" ")


_target_model_list = ["clip_resnet50", "ViT-B_32", "resnet50", "alexnet"]  # The model being dissected
_target_layers = "last_layer" # specialized for our work, more details can be seen in data_utils.py
# _d_probe = "nod_coco"  # dissect dataset
# _d_probe = "nod_subj01"
_d_probe = ["nod_coco", "nod_sub01", "nod_sub02", "nod_sub03",
"nod_sub04", "nod_sub05", "nod_sub06", "nod_sub07", "nod_sub08", "nod_sub09"]

current_model_id = 0 # 指定模型

_target_model = _target_model_list[current_model_id]

_concept_set = ''
_device = config.get("DEVICE", 'device')
# where to save activation and where to save visualize output
# _activation_dir = root_path + "/result/NSD_DissectActivation/{}Activation/subj0{}_{}_selective_concept_activations_SELECTIVE_ROI_{}".format(_target_model, current_subject, _target_model, roi_list[current_roi]) 
_activation_dir = root_path + "/result/features/NOD"

print("root_path:", root_path)

if not os.path.exists(_activation_dir):
    os.makedirs(_activation_dir)

_activation_root = "/".join(_activation_dir.split('/')[: -1])


if not os.path.exists(_activation_root):
    os.makedirs(_activation_root)

# loading Voxal weights


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default=_target_model, 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default=_target_layers,
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used
                          visual.bn1,visual.avgpool,visual.layer1.2.bn3,visual.layer2.3.bn3,visual.layer3.5.bn3,visual.layer4.2.bn3,visual.attnpool""")
parser.add_argument("--d_probe", type=str, default=_d_probe, 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden","floc"])
parser.add_argument("--concept_set", type=str, default=_concept_set, help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default=_device, help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default=_activation_dir, help="where to save activations")
# parser.add_argument("--result_dir", type=str, default=_result_dir, help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])
parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")
    

for d_probe in _d_probe:    
    save_features(clip_name = args.clip_model, target_name = args.target_model, 
                           target_layers = args.target_layers, d_probe = d_probe, 
                           batch_size = args.batch_size, 
                           device = args.device, pool_mode=args.pool_mode, 
                           save_dir = args.activation_dir,
                           weights = None)
    

