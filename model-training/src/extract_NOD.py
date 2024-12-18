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
import shutil

import configparser

config = configparser.ConfigParser()

config.read("config.cfg")

coco = config["DATASET"]["coco_full"]

NOD_path = config["NOD"]["DATA"] # data
ROOT = config["SAVE"]["ROOT"]
support_path = ROOT + "/result/supportfiles"

def prepare_imagenet_data(dataset_root, sub_names=None, support_path=support_path):
    # define path
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    nifti_path = f'{dataset_root}'
    if not sub_names:
        sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') ] )
    n_class = 1000
    num_ses, num_run, num_trial = 4, 10, 100
    vox_num = 59412
    # for each subject
    for sub_idx, sub_name in enumerate(sub_names):
        print(sub_name)
        label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
        # check whether label exists, if not then generate
        if not os.path.exists(label_file):
            sub_events_path = pjoin(nifti_path, sub_name)
            df_img_name = []
            # find imagenet task
            imagenet_sess = [_ for _ in os.listdir(sub_events_path) if ('imagenet' in _) and ('05' not in _)]
            imagenet_sess.sort()# Remember to sort list !!!
            # loop sess and run
            for sess in imagenet_sess:
                for run in np.linspace(1,10,10, dtype=int):
                    # open ev file
                    events_file = pjoin(sub_events_path, sess, 'func',
                                        '{:s}_{:s}_task-imagenet_run-{:02d}_events.tsv'.format(sub_name, sess, run))
                    tmp_df = pd.read_csv(events_file, sep="\t")
                    df_img_name.append(tmp_df.loc[:, ['trial_type', 'stim_file']])
            df_img_name = pd.concat(df_img_name)
            df_img_name.columns = ['class_id', 'image_name']
            df_img_name.reset_index(drop=True, inplace=True)
            # add super class id
            superclass_mapping = pd.read_csv(pjoin(support_path, 'superClassMapping.csv'))
            superclass_id = superclass_mapping['superClassID'].to_numpy()
            class_id = (df_img_name.loc[:, 'class_id'].to_numpy()-1).astype(int)
            df_img_name = pd.concat([df_img_name, pd.DataFrame(superclass_id[class_id], columns=['superclass_id'])], axis=1)
            # make path
            if not os.path.exists(support_path):
                os.makedirs(support_path)
            df_img_name.to_csv(label_file, index=False)
            print(f'Finish preparing labels for {sub_name}')
        # load sub label file
        label_sub = pd.read_csv(label_file)['class_id'].to_numpy()
        label_sub = label_sub.reshape((num_ses, n_class))
        # define beta path
        beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
        if not os.path.exists(beta_sub_path) or True:
            # extract from dscalar.nii
            beta_sub = np.zeros((num_ses, num_run*num_trial, vox_num))
            for i_ses in range(num_ses):
                for i_run in range(num_run):
                    run_name = f'ses-imagenet{i_ses+1:02d}_task-imagenet_run-{i_run+1}'
                    beta_data_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                    beta_sub[i_ses, i_run*num_trial : (i_run + 1)*num_trial, :] = np.asarray(nib.load(beta_data_path).get_fdata())
            # save session beta in ./supportfiles
            np.save(beta_sub_path, beta_sub)

            if not os.path.exists(ROOT+"/result/features/NOD/"):
                os.makedirs(ROOT+"/result/features/NOD/")

            np.save(ROOT+"/result/features/NOD/"+f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy', beta_sub)


def prepare_coco_data(dataset_root, sub_names=None, support_path=support_path,clean_code='hp128_s4'):
    # change to path of current file
    os.chdir(os.path.dirname(__file__))
    # define path
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    # Load COCO beta for 10 subjects
    if not sub_names:
        sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') and int(i[-2:])<=9])
    num_run = 10
    n_class = 120
    clean_code = 'hp128_s4'
    for _, sub_name in enumerate(sub_names):
        sub_data_path = f'{support_path}/{sub_name}_coco-beta_{clean_code}_ridge.npy'
        if not os.path.exists(sub_data_path) or True:
            # extract from dscalar.nii
            run_names = [ f'ses-coco_task-coco_run-{_+1}' for _ in range(num_run)]
            sub_beta = np.zeros((num_run, n_class, 59412))
            for run_idx, run_name in enumerate(run_names):
                beta_sub_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                sub_beta[run_idx, :, :] = np.asarray(nib.load(beta_sub_path).get_fdata())
            # save session beta in ./supportfiles
            np.save(sub_data_path, sub_beta)

            if not os.path.exists(ROOT+"/result/features/NOD/"):
                os.makedirs(ROOT+"/result/features/NOD/")

            np.save(ROOT+"/result/features/NOD/"+f'{sub_name}_coco-beta_{clean_code}_ridge.npy', sub_beta)

subs = ['sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09']
clean_code = 'hp128_s4'

print("prepare data")
prepare_imagenet_data(NOD_path, sub_names=subs)
print("imagenet data finish")
prepare_coco_data(NOD_path,  sub_names=subs)
print("coco data finish")

subs=['1','2','3','4','5','6','7','8','9']

for sub in subs:
    path = support_path + '/sub-0' + sub + '_imagenet-label.csv'
    save_path = ROOT + '/result/data_imagenet/sub-0'+sub+'/sub-0'+sub+'_data/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = pd.read_csv(path)
    image_name = np.array(images['image_name'])
    
    n = 0
    for name in image_name:
        image_path = NOD_path + '/stimuli/' + name
        n_0 = 4 - len(str(n))
        new_path = save_path + n_0 * '0' + str(n) + '.JPEG'
        shutil.copy(image_path, new_path)
        n += 1

images=pd.read_csv(support_path + '/coco_images.csv')
image_name=np.array(images['img'])
coco_full=os.listdir(coco + '/images')

save_path = ROOT + '/result/data_coco/NOD_coco/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

n=0
for name in image_name:
    while name[0]=='0':
        name=name[1:]
    if name in coco_full:
        n_0=3-len(str(n))
        image_path= coco + '/images/'+name
        new_path=save_path+n_0*'0'+str(n)+'.JPEG'
        shutil.copy(image_path,new_path)
        n+=1
    else:
        print('Error!')
        continue
# print(n)