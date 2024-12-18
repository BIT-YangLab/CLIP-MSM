import cv2
import os
import pandas as pd
import numpy as np
import configparser
config = configparser.ConfigParser()
root_path = os.path.abspath("../")
current_path = os.path.abspath("./")

config.read(current_path + "/config,cfg")

def get_target_label(data_root, target_label, image_root, subj_number, roi_name, image_save_root, number):
    csv_data = pd.read_csv(data_root)
    csv_data = csv_data.values
    new_list = []
    for i in range(csv_data.shape[0]):
        if csv_data[i][2] == target_label:
            new_list.append((csv_data[i][1], csv_data[i][3]))
    new_list.sort(reverse=True, key=lambda x: x[1])
    select_info = new_list[: number]
    all_image = []
    for each in select_info:
        index = str(each[0])
        for i in range(4 - len(index)):
            index = "0{}".format(index)
        target_image_root = image_root.format(roi_name, subj_number, index)
        voxal_image = cv2.imread(target_image_root)
        save_image = voxal_image[:, :227, :]
        all_image.append(save_image)
    output_image = np.concatenate(all_image, axis=1)
    img_save_root = "/".join(image_save_root.split("/")[: -1])
    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)
    cv2.imwrite(image_save_root, output_image)

model_list = ["ViT-B_32", "clip_resnet50", "resnet50", "alexnet"]
target_label_list = ['faces', 'bodies', 'places', 'words', 'food']
save_root = root_path + "/result/_{}_topImage/{}/Subj0{}"
save_name = "/{}.jpg"
all_number = 4
brain_area = ['ffa', "eba", "rsc", "vwfa", "food"]

for model in model_list:
    for i in range(1, 9):
        for j, each in enumerate(brain_area):
            image_save_name = save_root.format(all_number, model, i) + save_name.format(each)
            data_root = config.get("DISCRIPTIONS", model).format(each, i)
            target_label = target_label_list[j]
            image_root = config.get("DISSECTIMAGEROOT", model)
            get_target_label(data_root, target_label, image_root, i, each, image_save_name, all_number)

        