import pandas as pd
import os

import configparser

config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"] + "/result"

target_model = 'clip_resnet50'
model_path = '{}_last_layer_coco_full_SELECTIVE_ROI_full_'.format(target_model)
model_name = target_model

for subject in range(1,10):
    # hard-clip 
    dissect_path = '%s/NSD_DissectResult/%sResult/%sSubj0%d'%(target_model, model_path, subject)
    file_path = '%s/%s'%(dissect_path, model_name)
    # Read csv
    df = pd.read_csv('%s/tally.csv'%file_path)

    # Modify the unit column to remove the 'last_layer 'and Spaces and keep only the numbers
    df['unit'] = df['unit'].apply(lambda x: x.split()[-1])

    # Save to new csv file
    df.to_csv('%s/hard-clip-dissect.csv'%file_path, index=False)
    label = ['faces', 'bodies', 'places', 'words', 'food']
    save_dir = file_path + '/hard_label_index'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Created folder:", save_dir)
    else:
        print("Folder exists:", save_dir)
    output_file = os.path.join(save_dir, 'output.txt')
    with open(output_file, 'w') as f:
        for i in label:
            input_path = os.path.join(file_path, 'hard-clip-dissect.csv')
            df = pd.read_csv(input_path)
            df_filtered = df[(df['label'] == i)]
            df_sorted = df_filtered.sort_values(by='unit')
            output_csv = os.path.join(save_dir, 'hard-clip-dissect_hy_free_%s_index.csv' % (i))
            df_sorted[['unit']].to_csv(output_csv, index=False, header=False)
            f.write("Output file path: {}\n".format(output_csv))

        print("Result written to:", output_file)

    # soft-clip dissect
    # Read csv
    soft_path = '%s/output'%dissect_path
    df = pd.read_csv('%s/soft-dissect_loading.csv'%soft_path)

    save_path = '%s/loading_label'%soft_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Created Folder:", save_path)
    else:
        print("Folder exists:", save_path)

    # Loop through each column of data
    for col_name in df.columns:
        
        col_data = df[col_name]

        # Save the file to a CSV file using the column name
        csv_file_name = f'{save_path}/{col_name}.csv'
        col_data.to_csv(csv_file_name, index=False)
