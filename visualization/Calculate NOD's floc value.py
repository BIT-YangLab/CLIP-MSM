import os
import numpy as np

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"] + "/result"

labels = ['faces', 'bodies', 'places', 'words']
old_labels = ['Face', 'Body', 'Place', 'Character']
for i in range(1,10):
    for j in range(len(labels)):
        # Load floc_roi_value e.g. sub-01_Body.npy
        # Remember to run make_floc_response.py first
        floc_roi_value = np.load(ROOT + '/floc_response/sub-0%d_%s.npy'%(i,old_labels[j]))

        save_path = '%s/clip2brain/NOD_floc_beta_value/subj0%d'%(ROOT, i)

        min_val = np.min(floc_roi_value)
        max_val = np.max(floc_roi_value)
        if old_labels[j] == 'Place' or old_labels[j] == 'Character':
            print(min_val)
            print(max_val)
        # normalize
        normalized_floc_roi_value = (floc_roi_value - min_val) / (max_val - min_val)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save('%s/normalize_floc_roi_value_%s'%(save_path, labels[j]), normalized_floc_roi_value)