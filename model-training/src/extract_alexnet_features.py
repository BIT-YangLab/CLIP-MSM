import argparse
import copy

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

import torch
import torch.nn as nn

# import torchvision
from torchvision import transforms, utils, models

import torchextractor as tx

device = "cuda" if torch.cuda.is_available() else "cpu"
import configparser

config = configparser.ConfigParser()
config.read("config.cfg")
stimuli_dir = config["DATA"]["StimuliDir"]

ROOT = config["SAVE"]["ROOT"]

preprocess = transforms.Compose(
    [
        # transforms.Resize(375),
        transforms.ToTensor()
    ]
)

def extract_alexnet_avgpool_feature(saving=True):
    state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    model = models.alexnet(pretrained=False).to(device)
    model.load_state_dict(state_dict)

    model = tx.Extractor(model, "avgpool")

    # print("Extracting AlexNet features")
    output = list()
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            _, features = model(image)
            output.append(
                features["avgpool"].squeeze().data.cpu().numpy().flatten()
            )
            # which layer
    if saving:
        np.save("%s/convnet_alexnet_avgpool.npy" % feature_output_dir, output)
        # naming which layer
        pass

    return output

def extract_alexnet_last_layer_feature(saving=True):
    state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    model = models.alexnet(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    layer = "classifier.5"
    model = tx.Extractor(model, layer)

    # print("Extracting AlexNet features")
    output = list()
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            _, features = model(image)
            # print(features.keys())
            output.append(
                features[layer].squeeze().data.cpu().numpy().flatten()
            )
            # which layer
    if saving:
        np.save("%s/convnet_alexnet_last_layer.npy" % feature_output_dir, output)
        # naming which layer
        pass

    return output

def extract_resnet_last_layer_feature(cid=None, saving=True):
    model = models.resnet50(pretrained=True).to(device)
    # model = tx.Extractor(model, "avgpool")
    for name, layer in model.named_children():
        print(name)

def output_layer_name(path_to_alexnet_owt = "./checkpoints/alexnet-owt-7be5be79.pth"):
    model = models.alexnet(pretrained=False).to(device)
    model.load_state_dict(torch.load(path_to_alexnet_owt))
    for name, layer in model.named_children():
        print(name)

    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        x = input_tensor
        for name, layer in model.classifier.named_children():
            x = layer(x)
            print(f"Output of {name}: {x.shape}, {type(layer)}")

def alex_avg_PCA_1024(subj):
    try:
        f = np.load("%s/convnet_alexnet_avgpool.npy" % feature_output_dir)
    except FileNotFoundError as e:
        print(e)
        f = extract_alexnet_avgpool_feature()

    pca_shape = [1024]

    print("Running PCA")
    print("feature shape: ")
    print(f.shape)

    pca = PCA(n_components=min(f.shape[0], pca_shape[0]), svd_solver="auto")

    fp = pca.fit_transform(f)
    print("Feature %01d has shape of:" % subj)
    print(fp.shape)

    np.save("%s/convnet_alexnet_avgpool_PCA_%s.npy" % (feature_output_dir, pca_shape[0]), fp)

def alex_last_layer_PCA_512(subj):
    try:
        f = np.load("%s/convnet_alexnet_last_layer.npy" % feature_output_dir)
    except FileNotFoundError as e:
        print(e)
        f = extract_alexnet_last_layer_feature()

    pca_shape = [512]

    print("Running PCA")
    print("feature shape: ")
    print(f.shape)

    pca = PCA(n_components=min(f.shape[0], pca_shape[0]), svd_solver="auto")

    fp = pca.fit_transform(f)
    print("Feature %01d has shape of:" % subj)
    print(fp.shape)

    np.save("%s/convnet_alexnet_last_layer_PCA_%s.npy" % (feature_output_dir, pca_shape[0]), fp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    # parser.add_argument("--root", default=".", type=str)
    
    args = parser.parse_args()

    feature_dir = ROOT + "/result/features"
    output_dir = ROOT + "/result/output"
    if args.subj == 0:
        output_layer_name()
        extract_resnet_last_layer_feature()
    else:
        feature_output_dir = "%s/subj%01d" % (feature_dir, args.subj)
        all_coco_ids = np.load(
            "%s/coco_ID_of_repeats_subj%02d.npy" % (output_dir, args.subj)
        )
        
        extract_alexnet_avgpool_feature()
        alex_avg_PCA_1024(args.subj)
        # extract_alexnet_last_layer_feature()
        # alex_last_layer_PCA_512(args.subj)
