import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import data_utils
import sys
import torchextractor as tx
from sklearn.decomposition import PCA


PM_SUFFIX = {"max":"_max", "avg":"","test":""}



def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        print("max")
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='test':
        def hook(model, input, output):
#             print("test hook")
            if len(output.shape)==4: #CNN layers
#                 print("test 4")
#                 print("output: ",output[:,:,0,0].shape)
#                 print("output: ",output[:,:,0,0])
#                 print(xyz)
                outputs.append(output[:,:,5,5].clone())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook


def get_weighted_activation(outputs, mode, weights):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
#                 print(type(output))
#                 print(output.shape)
#                 print(torch.einsum('aj,ijkm->iakm',wei,output).shape)
#                 print("mean: ",torch.einsum('aj,ijkm->iakm',wei,output).mean(dim=[2,3]).shape)
#                 print("mean: ",output.mean(dim=[2,3]).shape)
#                 print(abc)
                outputs.append(torch.einsum('ijkm,ja->iakm',output,weights).mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    
    return hook


def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir, roi=None):
    if roi is not None:
        target_save_name = "{}/{}_{}_{}_{}.pt".format(save_dir, d_probe, target_name, target_layer+"_"+roi,
                                                     PM_SUFFIX[pool_mode])
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', weights = None):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    print("save_target_activations")
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        print("all saved")
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        if weights is None:
            command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
            hooks[target_layer] = eval(command)
        else:
            command = "target_model.{}.register_forward_hook(get_weighted_activation(all_features[target_layer], pool_mode, weights))".format(target_layer)
            hooks[target_layer] = eval(command)

        
        
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
#     print(abc)
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def alex_last_layer_PCA_512(last_feature):
    pca_shape = [512]

    pca = PCA(n_components=min(last_feature.shape[0], pca_shape[0]), svd_solver="auto")

    fp = pca.fit_transform(last_feature)
    return fp


def get_last_layer_feature(dataset,target_save_name,batch_size = 1000,device = "cuda",model_name='RN50', weights=None, target_model=None, target_preprocess=None, target_name=None):
#     print("get_last_layer")
#     print("test save_name: ",target_save_name.format("last_layer"))
#     print("test: ",weights is None)
    if target_model is None:
        model,preprocess=clip.load(model_name,device=device)
    else:
        model = target_model
    
    all_features=[]
    if os.path.exists(target_save_name.format("last_layer")):
        return
    
    if target_name is not None and "alexnet" in target_name:
        layer = "classifier.5"

        save_info_list = []

        model = tx.Extractor(model, layer)
        with torch.no_grad():

            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                images = images.to(device)
                _, features = model(images)
                last_features = features[layer].to('cpu')
                last_features = last_features.view(last_features.shape[0], -1)
                save_info_list.append(last_features)
            save_info_tensor = torch.cat(save_info_list, dim=0)
            pca_result = alex_last_layer_PCA_512(save_info_tensor.numpy())


            for i, (images, labels) in tqdm(enumerate(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True))):
                # images = images.to(device)
                # _, features = model(images)
                    
                if weights is not None:
                    # last_features = features[layer].to('cpu').numpy()
                    # last_features = torch.from_numpy(last_features).view(last_features.shape[0], 256 * 6 * 6).to(device) @ weights
                    # all_features.append(last_features.cpu().data.numpy())
                    new_features = torch.from_numpy(pca_result[i * batch_size: i * batch_size + images.shape[0]]).to(device) @ weights
                    all_features.append(new_features.cpu().numpy())
                else:
                    all_features.append(
                        features[layer].data.cpu().numpy().flatten()
                    )
                    raise RuntimeError()
        all_features=np.concatenate(all_features)
        all_features=torch.from_numpy(all_features)
        
        print("last layer shape: ",all_features.shape)
        torch.save(all_features,target_save_name.format("last_layer"))

    elif target_name == "resnet50":
        layer = "avgpool"
        model = tx.Extractor(model, layer)
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                images = images.to(device)
                _, features = model(images)
                if weights is not None:
                    flattened_features = features[layer].view(features[layer].shape[0], weights.shape[0])
                    last_features = flattened_features @ weights
                    all_features.append(last_features.cpu().data.numpy())
                else:
                    all_features.append(
                        features[layer].data.cpu().numpy().flatten()
                    )
                    raise RuntimeError()
        all_features=np.concatenate(all_features)
        all_features=torch.from_numpy(all_features)
        
        print("last layer shape: ",all_features.shape)
        torch.save(all_features,target_save_name.format("last_layer"))

    elif 'clip' in target_name or 'ViT' in target_name:

        
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                images=images.to(device)
                images_features=model.encode_image(images)
                
                images_features=images_features.to(dtype=torch.float32)
                if weights is not None:        
                    images_features=images_features @ weights
                all_features.append(images_features.cpu().data.numpy())
        all_features=np.concatenate(all_features)
        all_features=torch.from_numpy(all_features)
        
        print("last layer shape: ",all_features.shape)
        torch.save(all_features,target_save_name.format("last_layer"))
    

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir, weights=None):
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    #ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_layers==['last_layer']:
        get_last_layer_feature(data_t,target_save_name,batch_size=batch_size,device=device,weights=weights, target_model=target_model, target_preprocess=target_preprocess, target_name=target_name)
        print("1")
    else: 
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode,weights)
        print("2")
    print("test save_activations")
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda"):
    
#     print("text_save_name: ",text_save_name)
#     print("target_save_name: ",target_save_name)
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    print("if size: ",image_features.shape)
    print("tf size: ",text_features.shape)
    del image_features, text_features
    torch.cuda.empty_cache()
    print("clip size: ",clip_feats.shape)
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    print("test target_feats")
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    print("test similarity 2")
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

    
    