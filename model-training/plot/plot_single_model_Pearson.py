# Mean and SD of Pearson on single model across sessions
# n_sessions: [5, 10, 15, 20, 25, 30, 34]
# Models:
# - Clip-Visual-ResNet  Sub 1-8
# - Clip-ViT            Sub 1-8
# - ConvNet-ResNet      Sub 1-8
# - ConvNet-AlexNet     Sub 1-8
# - RN50-NOD            Sub 1-9

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

root = config['SAVE']['ROOT']
output_dir = root + "/result/output"
figures_dir = root + "/result/figures"
features_dir = root + "/result/features"

models = [
    {"model": "clip_vit", "model_name": "CLIP_ViT", "Subj": "range(1,9)", "roi": "SELECTIVE_ROI"},
    {"model": "clip_visual_resnet", "model_name": "CLIP_RN50", "Subj": "range(1,9)", "roi": "SELECTIVE_ROI"},
    {"model": "convnet_resnet", "model_name": "Resnet", "Subj": "range(1,9)", "roi": "SELECTIVE_ROI"},
    {"model": "convnet_alexnet", "model_name": "Alexnet", "Subj": "range(1,9)", "roi": "SELECTIVE_ROI"},
    {"model": "rn50-NOD-new", "model_name": "rn50 on NOD", "Subj": "range(1,10)", "roi": "whole_brain"}
]

for item in models:
    model = item["model"]
    model_name = item["model_name"]
    roi = item["roi"]
    Subj = eval(item["Subj"])

    save_dir = figures_dir + "/%s_%s" % (model, roi)

    for subj in Subj:

        save_root = "%s/subj%02d" % (save_dir, subj)

        n_sessions = [5, 10, 15, 20, 25, 30, 34]
        mean_r = []
        std_r = []
        for n_session in n_sessions:
            
            pfile = "%s/encoding_results/%s_%s/subj%d/%d_session/corr_%s_%s.p" % (output_dir, model, roi, subj, n_session, model, roi)
            with open(pfile, 'rb') as f:
                corrs = pickle.load(f)

            corrs = np.array(corrs)
            r = corrs[:, 0]

            
            mean_r.append(np.mean(r))
            std_r.append(np.std(r))


        print("mean:", mean_r)
        print("std:", std_r)

        plt.errorbar(n_sessions, mean_r, yerr=std_r, fmt='-o',ecolor='red', elinewidth=2, capsize=5, label='Mean R with Std Dev')

        plt.legend()

        plt.xlabel('Session Number')
        plt.ylabel('Pearson Value')
        plt.title('Mean Pearson and Standard Deviation for Different Sessions On %s' % model_name)

        plt.grid(True)

        plt.show()

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.savefig(save_root+"/Pearson_%s_%s.png" % (model, roi))
        plt.cla()