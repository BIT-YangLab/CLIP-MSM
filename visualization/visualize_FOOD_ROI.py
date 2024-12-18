import os
import numpy as np
def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p
from visualize_in_pycortex_FOOD_ROI import make_volume
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
            mask_with_significance=False,
            measure="rsq",
            sig_method="fdr",
        )

        from save_3d_views import save_3d_views1
        for k, v in volumes.items():
                root = "%s/clip2brain/figures/3d_views/subj%s" % (ROOT, i)
                if not os.path.exists(root):
                    os.makedirs(root)
                _ = save_3d_views1(
                    v,
                    root,
                    k,
                    list_views=["lateral", "bottom", "back"],
                    list_surfaces=["inflated"],
                    with_labels=True,
                    size=(1024 * 4, 768 * 4),
                    trim=True,
                )
                import pdb
                pdb.set_trace()

