import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
import configparser
config = configparser.ConfigParser()
root_path = os.path.abspath("../")
current_path = os.path.abspath("./")

config.read(current_path + "/config,cfg")

subj_number = [i for i in range(1, 9)]
brain_roi_list = ['ffa', 'eba', 'rsc', 'vwfa', 'food']
model_list = ['clip_resnet', 'ViT-B_32', 'resnet50', 'alexnet']
label = ['food','faces','places','bodies','words']
color_list = [(240 / 255, 128 / 255, 128 / 255), (144 / 255, 238 / 255, 144 / 255), (255 / 255, 215 / 255, 0), (100 / 255, 149 / 255, 237 / 255), (255 / 255, 165 / 255, 0)]
data_root_list = [
    config.get("SOFTDISSECTROOT", key) for key in config.options("SOFTDISSECTROOT")
]

data_root_list = []


save_root = root_path + "/result/png_bar/{}/subj{}"
save_name = "/{}-roi-barfig.png"
for i, model in enumerate(model_list):
    for subj_id in subj_number:
        current_save_root = save_root.format(model, subj_id)
        if not os.path.exists(current_save_root):
            os.makedirs(current_save_root)
        for k, roi in enumerate(brain_roi_list):
            data_root = data_root_list[i]
            data_root = data_root.format(roi, subj_id)
            csv_data = pd.read_csv(data_root).values
            plot_info = [(j, 0) for j in range(5)]
            for j in range(csv_data.shape[0]):
                max_id = np.argmax(csv_data[j])
                plot_info[max_id] = (plot_info[max_id][0], plot_info[max_id][1] + 1)
            
            plot_info.sort(reverse=True, key=lambda x: x[1])
            plot_labels = [label[k] for (k, _) in plot_info]
            plt.figure(figsize=(10, 12))
            plt.bar(plot_labels, [info[1] for info in plot_info], color=color_list[k], width=0.8)
            plt.xticks(rotation=45, ha='right', va='top')
            
            current_save_name = current_save_root + save_name.format(roi)
            plt.yticks([0, int(plot_info[0][1] / 2), plot_info[0][1]], ["0", str(int(plot_info[0][1] / 2)), str(plot_info[0][1])])
            plt.xticks(fontsize=70)
            plt.yticks(fontsize=55)
            plt.subplots_adjust(top=0.95, bottom=0.25, left=0.2, right=0.95)
            plt.ylim(0, plot_info[0][1])
            plt.box(False)
            plt.savefig(current_save_name, dpi=720)
            plt.close()



