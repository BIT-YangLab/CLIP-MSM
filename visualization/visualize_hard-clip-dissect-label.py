import os
import numpy as np
def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p
from visualize_in_pycortex_hard_clip_dissect_label import make_volume
import cortex
import nibabel as nib
import configparser
config = configparser.ConfigParser()
config.read("config.cfg")
PPdataPath = config["DATA"]["PPdataPath"]
ROOT = config["SAVE"]["ROOT"] + "/result"
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
        # load ViT-B_32_last_layer_coco_full_SELECTIVE_ROI_full_Subj0%d, please specify the path
        loading_path = 'YourPath/ViT-B_32_last_layer_coco_full_SELECTIVE_ROI_full_Subj0%d'%i
        import matplotlib.colors as mcolors
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        
        colors = ['white'] # , 'lightgreen', 'yellow', 'cornflowerblue', 'orange'
        cmap = plt.cm.colors.ListedColormap(colors)
        volumes = {}
        volumes["clip_visual_resnet R^2"] = make_volume(
            loading_path=loading_path,
            label=label,
            subj=i,
            roi=roi,
            model=model,
            cmap=cmap,
            mask_with_significance=False,
            measure="rsq",
            sig_method="fdr",
        )

        roi_list = ["RSC", "PPA", "OPA", "EBA", "FFA-1", "FFA-2", "OFA", "VWFA-1"]
        for k in volumes.keys():
                print(i)
                file_path = "%s/clip2brain/NSD_ViT-B_32_hard_dissect_score_visualized"%(ROOT)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                filename = "%s/clip2brain/NSD_ViT-B_32_hard_dissect_score_visualized/NSD_ViT-B_32hard_clip_dissect_score_visualized_subj0%d_%s"%(ROOT, i,label)
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
                    dpi=3000
                )