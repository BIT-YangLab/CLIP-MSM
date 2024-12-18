import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
import os

root_path = os.path.abspath("../")
current_path = os.path.abspath("./")
main_path = os.path.abspath("../../")

csv_save_root = main_path + "/result/ResultAnalysis/NSD_coherrence.csv"
person_number = 8
color_list = [(240 / 255, 128 / 255, 128 / 255), (144 / 255, 238 / 255, 144 / 255), (255 / 255, 215 / 255, 0), (100 / 255, 149 / 255, 237 / 255), (255 / 255, 165 / 255, 0)]
surface_color_list = [(240 / 255, 128 / 255, 128 / 255, 0), (144 / 255, 238 / 255, 144 / 255, 0), (255 / 255, 215 / 255, 0, 0), (100 / 255, 149 / 255, 237 / 255, 0), (255 / 255, 165 / 255, 0, 0)]
new_dissect_labels: list = ['FFA', 'EBA', 'RSC', 'VWFA', 'FOOD']
dissect_name_labels: list = ['Faces', 'Bodies', 'Places', 'Words', 'Food']     
place_index = [2, 4, 0, 1, 3]
if __name__ == "__main__":
    csv_data = pd.read_csv(csv_save_root, header=None)
    key_index = [1, 4, 7, 10, 13]
    labels = [each for each in new_dissect_labels]
    labels = [labels[i] for i in place_index]
    name_labels = [each for each in dissect_name_labels]
    name_labels = [name_labels[i] for i in place_index]
    data = []
    for i in range(1, csv_data.shape[0]):
        data.append([csv_data.iloc[i][each] for each in key_index])
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = eval(data[i][j])
            except:
                data[i][j] = 0
    print(type(data[0::2]))
    soft_data_numpy = torch.tensor(data[0::2]).numpy()
    hard_data_numpy = torch.tensor(data[1::2]).numpy()
    soft_data_numpy = soft_data_numpy.T
    hard_data_numpy = hard_data_numpy.T
    soft_data_numpy = [soft_data_numpy[i] for i in place_index]
    hard_data_numpy = [hard_data_numpy[i] for i in place_index]
    color_list = [color_list[i] for i in place_index]
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, label in enumerate(labels):
        f1 = ax.boxplot(soft_data_numpy[i], positions=[i * 1.5 - 0.3], widths=0.4, showfliers=False, patch_artist=True,
                        boxprops={'color':color_list[i], 'linewidth': 3, "hatch": "////"}, whiskerprops={'color':color_list[i], 'linewidth': 1}, 
                        capprops={'color':color_list[i], 'linewidth': 1}, medianprops={'color':color_list[i], 'linewidth': 1}
                        )
        f2 = ax.boxplot(hard_data_numpy[i], positions=[i * 1.5 + 0.3], widths=0.4, showfliers=False, patch_artist=True,
                        boxprops={'color':color_list[i], 'linewidth': 3, "hatch": "xxxx"}, whiskerprops={'color':color_list[i], 'linewidth': 1}, 
                        capprops={'color':color_list[i], 'linewidth': 1}, medianprops={'color':color_list[i], 'linewidth': 1}
                        )


        for box in f1['boxes']:
            box.set_facecolor(surface_color_list[i])
        for box in f2['boxes']:
            box.set_facecolor(surface_color_list[i])
        tstatistic, pvalue = stats.ttest_rel(soft_data_numpy[i], hard_data_numpy[i])
        print(tstatistic)
        print(pvalue)
        print("")
        if pvalue > 0.05:
            write_info = "n.s."
        elif pvalue > 0.001:
            write_info = "*"

        else:
            write_info = "**"
        y_max = max([max(soft_data_numpy[i]), max(hard_data_numpy[i])])
        y_min = min([min(soft_data_numpy[i]), min(hard_data_numpy[i])])

        ax.plot([i * 1.5 - 0.3, i * 1.5 + 0.3], [y_max + (y_max - y_min) / 10, y_max + (y_max - y_min) / 10], linewidth=1, color='k')
        plt.plot([i * 1.5 - 0.3, i * 1.5 - 0.3], [y_max + (y_max - y_min) / 10, y_max + (y_max - y_min) / 10 - (y_max - y_min) / 20],
                 linewidth=1, color='k')
        plt.plot([i * 1.5 + 0.3, i * 1.5 + 0.3], [y_max + (y_max - y_min) / 10, y_max + (y_max - y_min) / 10 - (y_max - y_min) / 20],
                 linewidth=1, color='k')
        plt.text( i * 1.5 - 0.1, y_max + (y_max - y_min) / 9, write_info, fontsize=10, color='black')
        plt.ylim((0.1, 0.8))
    f3 = ax.boxplot(hard_data_numpy[0] - 100, positions=[4 * 1.5 + 0.3], widths=0.4, showfliers=False, patch_artist=True,
                    boxprops={'color': "black", 'linewidth': 1, "hatch": "////", "label": "CLIP-MSM", "alpha": 1}, whiskerprops={'color':color_list[i], 'linewidth': 1, "alpha": 1}, 
                    capprops={'color': "black", 'linewidth': 1, "alpha": 0}, medianprops={'color': "black", 'linewidth': 1, "alpha": 0}
                    )
    f4 = ax.boxplot(hard_data_numpy[0] - 100, positions=[4 * 1.5 + 0.3], widths=0.4, showfliers=False, patch_artist=True,
                    boxprops={'color': "black", 'linewidth': 1, "hatch": "xxxx", "label": "CLIP Dissection", "alpha": 1}, whiskerprops={'color':color_list[i], 'linewidth': 1, "alpha": 1}, 
                    capprops={'color': "black", 'linewidth': 1, "alpha": 1}, medianprops={'color': "black", 'linewidth': 1, "alpha": 1}
                    )   
    for box in f3['boxes']:
        box.set_facecolor("white")
    for box in f4['boxes']:
        box.set_facecolor("white")

    ax.set_frame_on(True)
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_linewidth(2)
    ax.set_xticks([i * 1.5 for i in range(len(name_labels))])
    plt.legend(loc="lower left", prop={'size': 12}, handlelength=4 * 0.5, handleheight=3 * 0.5,)
    ax.set_xticklabels(name_labels)
    ax.xaxis.labelcolor = color_list
    plt.savefig(main_path + '/NSD_boxplot.png', dpi=2080)
    
