# Mean and SD of RSQ on single model across sessions
# Subject: 1-8
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
        mean_rsq = []
        std_rsq = []
        for n_session in n_sessions:

            rsq_file = "%s/bootstrap" % output_dir
            filepath = "%s/%s_%s/subj%01d/%d_session/rsq_dist_%s_%s.npy" % (rsq_file, model, roi, subj, n_session, model, roi)

            rsq = np.load(filepath)

            rsq = np.mean(rsq, axis = 0)

            mean_rsq.append(np.mean(rsq))
            std_rsq.append(np.std(rsq))


        print("mean:", mean_rsq)
        print("std:", std_rsq)

        plt.errorbar(n_sessions, mean_rsq, yerr=std_rsq, fmt='-o',ecolor='red', elinewidth=2, capsize=5, label='Mean RSQ with Std Dev')

        plt.legend()

        plt.xlabel('Session Number')
        plt.ylabel('RSQ Value')
        plt.title('Mean RSQ and Standard Deviation for Different Sessions On %s' % model_name)

        plt.grid(True)

        plt.show()

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.savefig(save_root+"/mean_rsq_std_plot_%s_%s.png" % (model, roi))
        plt.cla()

    for subj in Subj:
        n_sessions = [5, 10, 15, 20, 25, 30, 34]
        mean_rsq = []
        for n_session in n_sessions:

            rsq_file = "%s/bootstrap" % output_dir
            filepath = "%s/%s_%s/subj%01d/%d_session/rsq_dist_%s_%s.npy" % (rsq_file, model, roi, subj, n_session, model, roi)

            rsq = np.load(filepath)

            rsq = np.mean(rsq, axis = 0)

            mean_rsq.append(np.mean(rsq))

        plt.plot(n_sessions, mean_rsq, label = "subj%s"%subj)

    plt.legend()
    plt.xlabel('Session Number')
    plt.ylabel('RSQ Value')
    plt.grid(True)
    plt.title('Mean RSQ analysis across Subj for Different Sessions On %s' % model_name)
    plt.savefig(save_dir+"/across_subj_analysis.png")