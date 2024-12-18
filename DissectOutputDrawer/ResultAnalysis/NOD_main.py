import torch
import numpy as np
import os
from dataclasses import dataclass
import os
import csv
import configparser
config = configparser.ConfigParser()
root_path = os.path.abspath("../")
current_path = os.path.abspath("./")
main_path = os.path.abspath("../../")

config.read(current_path + "/config,cfg")

soft_score_root: str = config.get("NODSCORE", "soft_score_root")                                 
hard_score_root: str = config.get("NODSCORE", "hard_score_root")                  
floc_score_root: str = config.get("NODSCORE", "floc_score_root")                                               
                     
person_number: int = 9                                                                                                                      
floc_number_for_each_dissect_label: int = 1                                                                                                 
root_end: str = '.npy'                                                                                                                      
dissect_labels: list = ['faces', 'bodies', 'places', 'words']                                                                       
new_dissect_labels: list = ['FFA', 'EBA', 'RSC', 'VWFA']
floc_labels: list = ['faces', 'bodies', 'places', 'words']    
csv_save_root = main_path + "/result/ResultAnalysis/NOD_coherrence.csv"
if not os.path.exists("/".join(csv_save_root.split("/")[: -1])):
    os.makedirs("/".join(csv_save_root.split("/")[: -1]))
    
color_list = [(240, 128, 128), (144, 238, 144), (255, 215, 0), (100, 149, 237), (255, 165, 0)]

soft_score_root_list: list = [soft_score_root.format(i) for i in range(1, person_number + 1)]
hard_score_root_list: list = [hard_score_root.format(i) for i in range(1, person_number + 1)]
floc_score_root_list: list = [floc_score_root.format(i) for i in range(1, person_number + 1)]



def get_coherrence(person_index: int, dissect_index: int):

    soft_data = np.load(soft_score_root_list[person_index] + dissect_labels[dissect_index] + root_end)
    hard_data = np.load(hard_score_root_list[person_index] + dissect_labels[dissect_index] + root_end)
    soft_coherrence_list = []
    hard_coherrence_list = []

    if dissect_index != len(dissect_labels):
        for _i in range(floc_number_for_each_dissect_label):
            floc_data = np.load(floc_score_root_list[person_index] + floc_labels[dissect_index * floc_number_for_each_dissect_label + _i] + root_end)
            soft_coherrence_list.append(np.corrcoef(soft_data, floc_data)[0, 1])
            hard_coherrence_list.append(np.corrcoef(hard_data, floc_data)[0, 1])

    return soft_coherrence_list, hard_coherrence_list
    


if __name__ == "__main__":

    csv_head = [" "] + floc_labels
    csv_data = []
    csv_data.append(csv_head)
    for i in range(person_number):
        person_soft_data_list = ["Person{} soft".format(i + 1)]
        person_hard_data_list = ["Person{} hard".format(i + 1)]
        for j in range(len(dissect_labels)):
            soft_coherrence_list, hard_coherrence_list = get_coherrence(person_index=i, dissect_index=j)
            person_soft_data_list = person_soft_data_list + soft_coherrence_list
            person_hard_data_list = person_hard_data_list + hard_coherrence_list
            if j != len(dissect_labels) - 1:
                print("person{}\t".format(i + 1), [floc_labels[j * floc_number_for_each_dissect_label + k] for k in range(floc_number_for_each_dissect_label)])
                print("Soft   \t", soft_coherrence_list)
                print("hard   \t", hard_coherrence_list)
            else:

                print("Soft   \t", soft_coherrence_list)
                print("hard   \t", hard_coherrence_list)
        csv_data.append(person_soft_data_list)
        csv_data.append(person_hard_data_list)
    with open(csv_save_root, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        