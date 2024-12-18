import os
import argparse
import datetime
import json
import pandas as pd
import torch
import numpy as np
import utils_old as utils
import data_utils
import similarity
from matplotlib import pyplot as plt
import settings
from visualize.report import generate_html_summary
from PIL import Image
import matplotlib.image as mpimg
import configparser
config = configparser.ConfigParser()
root_path = os.path.abspath("../")
current_path = os.path.abspath("./")

config.read(current_path + "/config.cfg")

_target_model_list = ["clip_resnet50"]  # The model being dissected
_target_layers = "last_layer" # specialized for our work, more details can be seen in data_utils.py
_d_probe = "coco_full"  # dissect dataset
current_model_id, current_subject, current_roi = torch.load(root_path + "/result/NOD_train_info.pt")
current_model_id = int(current_model_id)
current_subject = int(current_subject)
current_roi = int(current_roi)
_target_model = _target_model_list[current_model_id]
roi_list = ['ffa', 'eba', 'rsc', 'vwfa', 'full']

default_category_order = [
    'object',
    'scene',
    'part',
    'material',
    'texture',
    'color'
]

_concept_set = current_path + "/data/four_selective_concept.txt"
_device = config.get("DEVICE", 'device')

_activation_dir = root_path + "/result/NOD_DissectActivation/{}Activation/subj0{}_{}_selective_concept_activations_SELECTIVE_ROI_{}".format(_target_model, current_subject, _target_model, roi_list[current_roi]) 
_result_dir = root_path + "/result/NOD_DissectResult/%sResult/%s_%s_%s_SELECTIVE_ROI_%s_Subj0%d"%(_target_model, _target_model,_target_layers,_d_probe, roi_list[current_roi], current_subject)

_activation_root = "/".join(_activation_dir.split('/')[: -1])
_result_root = "/".join(_result_dir.split('/')[: -1])

if not os.path.exists(_activation_root):
    os.makedirs(_activation_root)
if not os.path.exists(_result_root):
    os.makedirs(_result_root)


print("start load")
weigths_root_list = [
    root_path + "/model-training/output/encoding_results/rn50-NOD-new_whole_brain/subj{}/34_session/weights_rn50-NOD-new_whole_brain.npy", 
]
weights = np.load(weigths_root_list[current_model_id].format(current_subject))
print("weights: ",weights.shape)
weights=weights.T

weights=torch.from_numpy(weights).to(_device)
weights=weights.to(dtype=torch.float32)
ffa=np.load(current_path + "/data/NOD_weights/NOD_face_sub-0{}_index.npy".format(current_subject))
eba=np.load(current_path + "/data/NOD_weights/NOD_body_sub-0{}_index.npy".format(current_subject))
rsc=np.load(current_path + "/data/NOD_weights/NOD_place_sub-0{}_index.npy".format(current_subject))
vwfa=np.load(current_path + "/data/NOD_weights/NOD_word_sub-0{}_index.npy".format(current_subject))
index=np.zeros(len(weights),dtype=bool)
roi_info_list = [ffa, eba, rsc, vwfa, 0]
if current_roi < 4:
    ir = roi_info_list[current_roi]
else:
    ir = ffa
for i in range(len(weights)):
    if current_roi < 4:
        if ir[i]>0:
            index[i]=True
    else:
        if ffa[i]>0 or eba[i]>0 or rsc[i]>0 or vwfa[i]>0 or food[i] > 0:
            index[i] = True
weights=weights[index][:]
weights=weights.T

print("weights: ",weights.shape)
print("finish load")


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default=_target_model, 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default=_target_layers,
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used
                          visual.bn1,visual.avgpool,visual.layer1.2.bn3,visual.layer2.3.bn3,visual.layer3.5.bn3,visual.layer4.2.bn3,visual.attnpool""")
parser.add_argument("--d_probe", type=str, default=_d_probe, 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden","floc"])
parser.add_argument("--concept_set", type=str, default=_concept_set, help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default=_device, help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default=_activation_dir, help="where to save activations")
parser.add_argument("--result_dir", type=str, default=_result_dir, help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")
    
    similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    
    utils.save_activations(clip_name = args.clip_model, target_name = args.target_model, 
                           target_layers = args.target_layers, d_probe = args.d_probe, 
                           concept_set = args.concept_set, batch_size = args.batch_size, 
                           device = args.device, pool_mode=args.pool_mode, 
                           save_dir = args.activation_dir,
                           weights = weights)
    
    outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    os.mkdir("{}".format(args.result_dir))
    os.mkdir("{}/output".format(args.result_dir))
    os.mkdir("{}/vis_result".format(args.result_dir))
    for target_layer in args.target_layers:
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names
        print("test save_names")
        similarities,target_feats = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True, device=args.device
        )

        # hard dissect：记录每个voxel similarity score得分最高的label
        top10_vals, top10_ids = torch.topk(similarities, k=4, dim=1)
        top_des=[]
        for top_neu_ids in top10_ids:
            #print(top_neu_ids.shape)
            des=[]
            for idx in top_neu_ids:
                #print(idx)
                des.append(words[idx])
            top_des.append(des)
        df_top10=pd.DataFrame(top_des,columns=['top1','top2','top3','top4'])

        df_top10.to_csv("{}/output/top5_{}.csv".format(args.result_dir,target_layer),index=False,encoding='utf-8')
        
        vals, ids = torch.max(similarities, dim=1)
        top_vals, top_ids = torch.topk(target_feats, k=5, dim=0)
        pil_data = data_utils.get_data(args.d_probe)
        f_name=[]
        le=len(top_ids[0])
        name_index = 4

        for i in range(le):
            name=[]
            for j in range(name_index):
                name.append(pil_data.imgs[top_ids[j,i]][0])
                #print(pil_data.imgs[top_ids[j,i]][0])
            name=np.array(name)
            f_name.append(name)
        f_name=np.array(f_name)

        test_id=0
    
        for i, top_id in enumerate(top_ids[:, test_id]):

            im, label = pil_data[top_id]

        descriptions = [words[int(idx)] for idx in ids]

        np.save("{}/output/filename_{}.npy".format(args.result_dir,target_layer),f_name)

        import copy
        # sm  = copy.deepcopy(similarities)
        import torch
        import torch.nn.functional as F
        
        
        # soft dissect：对每个voxel对所有label的similarity score进行softmax
        similarities = F.softmax(similarities  , dim=1)
        # similarities = torch.clamp(similarities, 1e-6, 1 - 1e-6)
        loading_sm = copy.deepcopy(similarities)
        loading_sm = loading_sm.cpu().numpy()
        loading = {}
        for j in range(len(words)):
            loading[words[j]] = loading_sm[:,j]
        import pandas as pd
        df = pd.DataFrame(loading, dtype='float64')
        df.to_csv("{}/output/soft-dissect_loading.csv".format(args.result_dir),index=False,encoding='utf-8')

        del similarities
        torch.cuda.empty_cache()
        
        descriptions = [words[int(idx)] for idx in ids]
        
        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())
        
    df = pd.DataFrame(outputs)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    save_path = "{}/{}".format(args.result_dir, args.target_model)
    os.mkdir(save_path)
    df.to_csv(os.path.join(save_path,"descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # to_tally   
    de=pd.read_csv("{}/{}/descriptions.csv".format(args.result_dir, args.target_model)) ###改这里
    length=len(de)

    for i in range(length):
        de['unit'][i]=de['layer'][i]+' '+str(de['unit'][i])
    
    de['label']=de['description']
    de['score']=de['similarity']
    de=de.drop('layer',axis=1)
    de=de.drop('description',axis=1)
    de=de.drop('similarity',axis=1)
    de['category']=default_category_order[current_roi]
    de.to_csv("{}/{}/tally.csv".format(args.result_dir, args.target_model),index=False)
    
    #to_result
    for target_layer in args.target_layers:
        tally_pd=pd.read_csv("{}/{}/tally.csv".format(args.result_dir, args.target_model))
        f_name=np.load("{}/output/filename_{}.npy".format(args.result_dir,target_layer))
        l=len(tally_pd)
        rets = [None] * l

        for i in range(l):
            data = {
                    'unit': tally_pd.iloc[i]['unit'],
                    'category': tally_pd.iloc[i]['category'],
                    'label': tally_pd.iloc[i]['label'],
                    'score': tally_pd.iloc[i]['score']
                    }
            rets[i]=data
        layer=target_layer
        generate_html_summary(layer=layer,output_folder="{}/vis_result".format(args.result_dir),tally_result=rets,top_name=f_name)
