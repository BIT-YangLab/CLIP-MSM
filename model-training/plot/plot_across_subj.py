# Comparative RSQ Analysis
# Subjects: 1-8
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

save_root = figures_dir + '/across_subj'

if not os.path.exists(save_root):
    os.makedirs(save_root)

model = ["clip_visual_resnet", "clip_vit", "convnet_resnet", "convnet_alexnet"]
model_name = ["CLIP_RN50", "CLIP_ViT", "Resnet", "Alexnet"]

Subj = range(1,9)

mean_rsq = {model_name[i]: [] for i in range(len(model_name))}
std_rsq = {model_name[i]: [] for i in range(len(model_name))}

n_session = 34

roi = "SELECTIVE_ROI"

for subj in Subj:
    for i, m in enumerate(model):
        rsq_file = "%s/bootstrap" % output_dir
        filepath = f"{rsq_file}/{m}_{roi}/subj{subj:01d}/{n_session}_session/rsq_dist_{m}_{roi}.npy"
        rsq = np.load(filepath)
        print(rsq.shape)
        rsq = np.mean(rsq, axis=0)
        print(rsq.shape)
        mean_rsq[model_name[i]].append(np.mean(rsq))
        std_rsq[model_name[i]].append(np.std(rsq))

fig, ax = plt.subplots()
for i, m_name in enumerate(model_name):
    ax.errorbar(Subj, mean_rsq[m_name], yerr=std_rsq[m_name], fmt='-o', ecolor=f'C{i}', elinewidth=2, capsize=5, label=m_name)
    
plt.legend()
plt.xlabel('Subj ID')
plt.ylabel('RSQ Value')
plt.title('Mean RSQ and Standard Deviation across Subj')
plt.grid(True)
plt.show()
plt.savefig("%s/across_subj.png" % save_root)
plt.cla()