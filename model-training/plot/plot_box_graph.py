# Boxplots for RSQ values with significance indicators
# Subject: 1-8
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

save_root = figures_dir + "/box-graph"

if not os.path.exists(save_root):
    os.makedirs(save_root)

model = ["clip_visual_resnet", "clip_vit", "convnet_resnet", "convnet_alexnet"]
model_name = ["CLIP_RN50", "CLIP_ViT", "Resnet", "Alexnet"]

n_session = 34
roi = "SELECTIVE_ROI"

Subj = range(1,9)

for subj in Subj:
    all_rsq = []
    for i, m in enumerate(model):
        rsq_file = "%s/bootstrap" % output_dir
        filepath = f"{rsq_file}/{m}_{roi}/subj{subj:01d}/{n_session}_session/rsq_dist_{m}_{roi}.npy"
        rsq = np.load(filepath)
        rsq = np.mean(rsq, axis=0)
        print(rsq.shape)
        all_rsq.append(rsq)

    fig, ax = plt.subplots()

    positions = range(len(model))

    for i, model_rsq in enumerate(all_rsq):
        ax.boxplot(model_rsq, positions=[i], widths=0.3, 
                showfliers=False,    
                medianprops={'color': f'C{i}', 'linewidth': 4}, 
                whiskerprops={'color': f'C{i}', 'linewidth': 4},  
                capprops={'color': f'C{i}', 'linewidth': 4},      
                boxprops={'color': f'C{i}',  'linewidth': 4})
        
    positions = np.arange(len(all_rsq))

    if subj != 6:
        x = [(0,3),(0,2),(1,3),(1,2),(2,3)]
        star = ["**","**","**","**","**"]
    # x = [(0,2),(0,3),(1,2),(1,3),(2,3)]
    else:
        x = [(0,3),(0,2),(1,3),(1,2)]
        star = ["**","**","**","**"]

    y = np.linspace(0.98, 0.72, len(x))


    for i in range(len(x)):
        x1, x2 = x[i]
        ax.plot([x1, x2], [y[i], y[i]], color = 'black', linewidth=1)
        ax.plot([x1, x1], [y[i]-0.018, y[i]], color = 'black', linewidth=1)
        ax.plot([x2, x2], [y[i]-0.018, y[i]], color = 'black', linewidth=1)
        ax.text((x1+x2)/2, y[i], star[i], horizontalalignment='center', fontsize=14)

    # ax.set_xticks(positions)
    ax.set_xticklabels(model_name)
    # ax.set_title('RSQ Boxplot for Different Models')
    # ax.set_xlabel('Models', fontsize=15)
    # ax.set_ylabel('RSQ Value', fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim(-0.05, 1.00)

    plt.tight_layout()
    # plt.show()

    ax.tick_params(labelbottom=False, labelleft=False, width=2)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.savefig(f"{save_root}/{subj:02d}.png")


    plt.cla()
    plt.close()