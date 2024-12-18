# model performance versus noise ceiling 

import numpy as np
import matplotlib.pyplot as plt
from util.scatter import scatter
import os

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

root = config['SAVE']['ROOT']
output_dir = root + "/result/output"
figures_dir = root + "/result/figures"
features_dir = root + "/result/features"

Subj = range(1,9)
Model = [
    "clip_visual_resnet_SELECTIVE_ROI",
    "clip_vit_SELECTIVE_ROI",
    "convnet_alexnet_SELECTIVE_ROI",
    "convnet_resnet_SELECTIVE_ROI",
]
for subj in Subj:
    for model in Model:
        # please specify
        noise_path = "YourPath/noise_ceilings/subj%02d/noise_ceiling.npy" % subj
        noise = np.load(noise_path)
        noise.shape

        print(noise.max())

        rsq_path = "%s/encoding_results/%s/subj%d/34_session/rsq_%s.p" % (output_dir, model, subj, model)
        rsq = np.load(rsq_path, allow_pickle=True)
        rsq.shape

        fdr_mask_path = "%s/ci_threshold/%s_34_session_fdr_p_subj%d.npy" % (output_dir, model, subj)
        fdr_mask = np.load(fdr_mask_path)[0].astype(bool)

        noise, rsq = noise[fdr_mask], rsq[fdr_mask]

        data = np.vstack((noise, rsq)).T

        save_path = "%s/noise_ceiling/%s" % (figures_dir, model)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        scatter(data, save = os.path.join(save_path, "%02d.png" % subj))