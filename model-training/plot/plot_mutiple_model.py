# Mean and SD of RSQ for models across sessions
# Subject: 1-8
# n_sessions: [5, 10, 15, 20, 25, 30, 34]
# Models:
# - Clip-Visual-ResNet
# - Clip-ViT
# - ConvNet-ResNet
# - ConvNet-AlexNet

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

save_root = figures_dir + "/4-models"

if not os.path.exists(save_root):
    os.makedirs(save_root)

model = ["clip_visual_resnet", "clip_vit", "convnet_resnet", "convnet_alexnet"]
model_name = ["CLIP_RN50", "CLIP_ViT", "Resnet", "Alexnet"]

n_sessions = [5, 10, 15, 20, 25, 30, 34]

roi = "SELECTIVE_ROI"

Subj = range(1,9)

for subj in Subj:
    mean_rsq = {model_name[i]: [] for i in range(len(model_name))}
    std_rsq = {model_name[i]: [] for i in range(len(model_name))}
    for i, m in enumerate(model):
        for n_session in n_sessions:
            if n_session == 0:
                mean_rsq[model_name[i]].append(0)
                std_rsq[model_name[i]].append(0)
            else:
                rsq_file = "%s/bootstrap" % output_dir
                filepath = f"{rsq_file}/{m}_{roi}/subj{subj:01d}/{n_session}_session/rsq_dist_{m}_{roi}.npy"
                rsq = np.load(filepath)
                rsq = np.mean(rsq, axis=0)
                mean_rsq[model_name[i]].append(np.mean(rsq))
                std_rsq[model_name[i]].append(np.std(rsq))

    # 均值和标准差
    for i, m_name in enumerate(model_name):
        print(f"{m_name} mean:", mean_rsq[m_name])
        print(f"{m_name} std:", std_rsq[m_name])

    fig, ax = plt.subplots()

    for i, m_name in enumerate(model_name):
        ax.plot(n_sessions, mean_rsq[m_name], label=m_name, marker='o', markersize=8, linewidth=4)
        # ax.errorbar(n_sessions, mean_rsq[m_name], yerr=std_rsq[m_name], fmt='-o', ecolor=f'C{i}', elinewidth=2, capsize=5, label=m_name)

    # plt.ylim(0, 0.4)
    
    # plt.xlabel('Training samples (×250)', fontsize = 15)
    # plt.ylabel('Model performance (R²)', fontsize = 15)
    # plt.title(f'Mean RSQ and Standard Deviation for Different Sessions on subj{subj:02d}')

    plt.ylim(0, 0.22)

    
    # plt.legend(fontsize=16, loc='lower right')
    plt.xticks(n_sessions, fontsize = 12)
    plt.grid(True, linewidth = 2)
    ax.tick_params(labelbottom=False, labelleft=False, width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # plt.show()

    plt.tight_layout()
    plt.savefig(f"{save_root}/subj{subj:02d}.png")
    plt.cla()