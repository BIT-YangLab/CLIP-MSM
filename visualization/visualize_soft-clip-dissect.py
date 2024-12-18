import os
import numpy as np
def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p
from visualize_in_pycortex_soft_clip_dissect import make_volume
import cortex
import nibabel as nib
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")
PPdataPath = config["DATA"]["PPdataPath"]
ROOT = config["SAVE"]["ROOT"]+ "/result"
subjects = 9
model = 'clip_visual_resnet'
roi = 'SELECTIVE_ROI' # floc-roi

labels = ['faces', 'bodies', 'places', 'words', 'food'] # faces, 'bodies', 'words', 'places', 'food'

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
        # Please specify path
        loading_path = 'YourPath/ViT-B_32_last_layer_coco_full_SELECTIVE_ROI_full_Subj0%d/output/loading_label'%i
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
            mask_with_significance=False,
            measure="rsq",
            sig_method="fdr",
        )

        roi_list = ["RSC", "PPA", "OPA", "EBA", "EarlyVis", "FFA-1", "FFA-2"]
        for k in volumes.keys():
                file_path = "%s/clip2brain/B_32_soft_clip_dissect_results"%(ROOT)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                filename = ROOT + "/clip2brain/NSD_ViT-B_32_soft_clip_dissect_results/soft_clip_dissect_subj0%d_%s"%(i,label)
                _ = cortex.quickflat.make_png(
                    filename,
                    volumes[k],
                    linewidth=3,
                    labelsize="20pt",
                    with_curvature=True,
                    recache=False,
                    roi_list=roi_list,
                    with_labels=False,
                    with_colorbar=False,
                )
