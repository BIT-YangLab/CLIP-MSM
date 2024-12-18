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

subs=['1','2','3','4','5','6','7','8','9']

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"] + "/result"
NOD_path = config["NOD"]["DATA"]
Mask_path = ROOT + '/mask'

for sub in subs:
    path = NOD_path + '/derivatives/ciftify/sub-0' + sub + '/results/ses-floc_task-floc/'
    save_path = ROOT + '/floc_response/'

    whole_brain = nib.load(path)
    whole_brain = whole_brain.get_fdata()
    mask = nib.load(Mask_path + '/floc_roi_sub' + sub + '.dlabel.nii')
    mask = mask.get_fdata()[0]
    mask = mask > 0

    whole_brain = whole_brain[:, mask]

    np.save(save_path + 'sub-0' + sub + '_Character.npy', whole_brain[0])
    np.save(save_path + 'sub-0' + sub + '_Body.npy', whole_brain[1])
    np.save(save_path + 'sub-0' + sub + '_Face.npy', whole_brain[2])
    np.save(save_path + 'sub-0' + sub + '_Place.npy', whole_brain[3])
    np.save(save_path + 'sub-0' + sub + '_Object.npy', whole_brain[4])