import os
import numpy as np
def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p
from visualize_in_pycortex_2 import make_volume
import cortex
import nibabel as nib
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")
PPdataPath = config["DATA"]["PPdataPath"]
ROOT = config["SAVE"]["ROOT"] + "/result"
subjects = 9
model = 'clip_visual_resnet' # clip_visual_resnet, clip_vit, convnet_resnet, convnet_alexnet, convnet_alexnet_last_layer_PCA_512
roi = 'SELECTIVE_ROI' # floc-roi

labels = ['faces'] # faces, 'bodies', 'words', 'places', 'food'
loading_path = 'clip_resnet50_last_layer_coco_full_FLOC_ROI_new_hy-free_all_voxels'

for label in labels:
    for i in range(1,9):
        rsq = np.load(
            "%s/output/bootstrap/%s_%s/subj%01d/34_session/rsq_dist_%s_%s.npy"%(ROOT, model, roi, i, model, roi)
        )
        fdr_p = fdr_correct_p(rsq)
        print(np.sum(fdr_p[1] < 0.05))
        if not os.path.exists("%s/output/ci_threshold" % (ROOT)):
            os.makedirs("%s/output/ci_threshold" % (ROOT))
        np.save(
            "%s/output/ci_threshold/%s_%s_34session_fdr_p_subj%01d.npy"
            % (ROOT, model, roi, i),
            fdr_p,
        )

        import matplotlib.colors as mcolors
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        volumes = {}
        volumes["clip_visual_resnet R^2"] = make_volume(
            loading_path=loading_path,
            label=label,
            subj=i,
            roi=roi,
            model=model,
            mask_with_significance=True,
            measure="rsq",
            sig_method="fdr",
        )
        roi_list = ["RSC", "PPA", "OPA", "EBA", "FFA-1", "FFA-2"]
        for k in volumes.keys():
                root = "%s/clip2brain/%s_%s_training_results" %(ROOT, model, roi)
                if not os.path.exists(root):
                    os.makedirs(root)
                filename = "%s/%s_%s_subj0%d"%(root, model, roi, i)
                _ = cortex.quickflat.make_png( 
                    filename,
                    volumes[k],
                    linewidth=3,
                    labelsize="20pt",
                    with_curvature=True,
                    recache=False,
                    roi_list=roi_list,
                    with_labels=True,
                    with_colorbar=False,
                    dpi=5000,
                )