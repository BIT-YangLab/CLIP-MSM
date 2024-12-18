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

import configparser

config = configparser.ConfigParser()

config.read("config.cfg")

NOD_path = config["NOD"]["DATA"] # data
ROOT = config["SAVE"]["ROOT"]
support_path = ROOT + "/result/supportfiles"

subs=['1','2','3','4','5','6','7','8','9']

for sub in subs:
    path = NOD_path + '/derivatives/ciftify/sub-0' + sub + '/results/ses-floc_task-floc/'
    if not os.path.exists(ROOT + '/result/mask'):
        os.makedirs(ROOT + '/result/mask')
    save_path = ROOT + '/result/mask/floc_roi_sub' + sub + '.dlabel.nii'
    
    floc = nib.load(path+'floc-faces.dlabel.nii')
    floc = np.asarray(floc.get_fdata())
    floc_roi=np.zeros(floc.shape)
    

    faces = nib.load(path + 'floc-faces.dlabel.nii')
    faces_data = np.asarray(faces.get_fdata())
    for v in range(len(faces_data[0])):
        if faces_data[0][v] != 0 and floc_roi[0][v] == 0:
            floc_roi[0][v] = 1

    bodies = nib.load(path + 'floc-bodies.dlabel.nii')
    bodies_data = np.asarray(bodies.get_fdata())
    for v in range(len(bodies_data[0])):
        if bodies_data[0][v] != 0 and floc_roi[0][v] == 0:
            floc_roi[0][v] = 2

    places = nib.load(path + 'floc-places.dlabel.nii')
    places_data = np.asarray(places.get_fdata())
    for v in range(len(places_data[0])):
        if places_data[0][v] != 0 and floc_roi[0][v] == 0:
            floc_roi[0][v] = 3

    words = nib.load(path + 'floc-words.dlabel.nii')
    words_data = np.asarray(words.get_fdata())
    for v in range(len(words_data[0])):
        if words_data[0][v] != 0 and floc_roi[0][v] == 0:
            floc_roi[0][v] = 4

    mem = faces.get_fdata().copy()
    for i in range(len(floc_roi[0])):
        mem[0][i] = floc_roi[0][i]

    header = faces.header.copy()
    mask = nib.Cifti2Image(mem, header)
    nib.save(mask, save_path)

