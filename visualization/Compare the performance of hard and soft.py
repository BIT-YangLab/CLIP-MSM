import os
import numpy as np
import nibabel as nib
import pandas as pd

# Get the score of all soft clip dissect
labels = ['faces', 'bodies', 'words', 'places', 'food']
target_model = 'clip_resnet50'
model_path = '{}_last_layer_coco_full_SELECTIVE_ROI_full_'.format(target_model)
model_name = target_model

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"] + "/result"

for subject in range(1,9):
    for label in labels:
        loading_path = '%s/NSD_DissectResult/%sResult/%sSubj0%d/output/loading_label'%(ROOT, target_model, model_path, subject)
        df = pd.read_csv('%s/%s.csv'%(loading_path, label))
        loading = df.to_numpy()
        loading = loading.reshape(len(loading))
        save_path = '%s/clip2brain/NSD_ViT-B_32_soft_dissect_score/subj0%d'%(ROOT,subject)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save('%s/NSD_ViT-B_32_soft_dissect_score_%s'%(save_path, label), loading)

# Get the score of all hard clip dissect
labels = ['faces', 'bodies', 'words', 'places', 'food']
for i in range(1,9):
    for label in labels:

        # Reads all results of hard dissect
        hard_label = '%s/NSD_DissectResult/%sResult/%sSubj0%d/%s'%(ROOT, starget_model, model_path, subject, model_name)
        df = pd.read_csv('%s/hard-clip-dissect.csv'%(hard_label))
        df_label = df[df['label'] == label]

        # Gets the voxel index for a hard label
        df_unit = df_label['unit']
        hard_unit = df_unit.to_numpy()
        hard_unit = hard_unit.reshape(len(hard_unit))

        # Save to ROI
        score = np.zeros(len(df),dtype=float)
        score[hard_unit] = 1
        save_path = '%s/clip2brain/NSD_ViT-B_32_hard_dissect_score_softmax/subj0%d'%(ROOT,i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save('%s/NSD_ViT-B_32_hard_dissect_score_softmax_%s'%(save_path, label), score)