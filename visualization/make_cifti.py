import numpy as np
import os
import time
import numpy as np
import subprocess
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio
import nibabel as nib
from scipy.stats import zscore
import nibabel as nib
import scipy.io as sio
from nibabel import cifti2

def norm(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

subs=['1','2','3','4','5','6','7','8','9']
cates=['bodies','faces','places','words']

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

NOD_path=config["NOD"]["DATA"]
ROOT = config["SAVE"]["ROOT"] + "/result"
Mask_path = ROOT + '/mask'

for sub in subs:
    for cat in cates:
        score_path = ROOT + '/clip2brain/NOD_ViT-B_32_hard_dissect_score_softmax/subj0' + sub + '/NOD_ViT-B_32_hard_dissect_score_softmax_' + cat + '.npy'
        save_path = ROOT + '/hard/sub0' + sub + '_' + cat + '.dscalar.nii'

        score = np.load(score_path)
        score = norm(score)
        print('sub ' + sub + ' score shape', score.shape)
        print('max: ', np.max(score))
        print('min: ', np.min(score))
        mask = nib.load(Mask_path + '/floc_roi_sub' + sub + '.dlabel.nii')
        mask = mask.get_fdata()[0]
        mem = np.zeros(mask.shape, dtype=float)

        locate = 0
        for i in range(mem.shape[0]):
            if mask[i] > 0:
                mem[i] = score[locate]
                locate += 1
            else:
                mem[i] = np.nan

        mem = mem[np.newaxis, :]
        dscalar = nib.load(NOD_path + '/derivatives/ciftify/sub-0' + sub + '/results/ses-imagenet01_task-imagenet_run-1/ses-imagenet01_task-imagenet_run-1_beta.dscalar.nii')
        header = dscalar.header.copy()
        axis = nib.cifti2.cifti2_axes.ScalarAxis(['score'])
        cif_header = nib.cifti2.cifti2.Cifti2Header.from_axes((axis, header.get_axis(1)))
        cif = nib.Cifti2Image(mem, cif_header)

        nib.save(cif, save_path)
