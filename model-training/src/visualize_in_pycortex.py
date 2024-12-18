"This scripts visualize prediction performance with pycortex."
import os
from nibabel.volumeutils import working_type
import numpy as np

import argparse
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm

from util.data_util import load_model_performance

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"]

def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals[~mask] = np.nan #
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def project_vols_to_mni(subj, vol):
    import cortex

    xfm = "func1pt8_to_anat0pt8_autoFSbbr"
    # template = "func1pt8_to_anat0pt8_autoFSbbr"
    mni_transform = cortex.db.get_mnixfm("subj%02d" % subj, xfm)
    mni_vol = cortex.mni.transform_to_mni(vol, mni_transform)
    mni_data = mni_vol.get_fdata().T
    return mni_data


def load_fdr_mask(OUTPUT_ROOT, model, fdr_mask_name, subj, roi, n_session):
    if type(fdr_mask_name) is list:
        sig_mask1 = np.load(
            "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
            % (OUTPUT_ROOT, fdr_mask_name[0], subj)
        )[0].astype(bool)
        sig_mask2 = np.load(
            "%s/output/ci_threshold/%s_fdr_p_subj%01d.npy"
            % (OUTPUT_ROOT, fdr_mask_name[1], subj)
        )[0].astype(bool)
        sig_mask = (sig_mask1.astype(int) + sig_mask2.astype(int)).astype(bool)
        return sig_mask
    elif fdr_mask_name is not None:
        model = fdr_mask_name
    try:
        sig_mask = np.load(
            "%s/output/ci_threshold/%s_%s_%d_session_fdr_p_subj%01d.npy" % (OUTPUT_ROOT, model, roi, n_session, subj)
        )[0].astype(bool) #notice
        return sig_mask
    except FileNotFoundError:  # hasn't run the test yet
        return None


def visualize_layerwise_max_corr_results(
    model, layer_num, subj=1, threshold=95, start_with_zero=True, order="asc"
):
    val_array = list()
    for i in range(layer_num):
        if not start_with_zero:  # layer starts with 1
            continue
        val_array.append(
            load_model_performance(
                model="%s_%d" % (model, i), output_root=OUTPUT_ROOT, subj=args.subj
            )
        )

    val_array = np.array(val_array)

    threshold_performance = np.max(val_array, axis=0) * (threshold / 100)
    layeridx = np.zeros(threshold_performance.shape) - 1

    i = 0 if order == "asc" else -1
    for v in tqdm(range(len(threshold_performance))):
        if threshold_performance[v] > 0:
            layeridx[v] = (
                int(np.nonzero(val_array[:, v] >= threshold_performance[v])[0][i]) + 1
            )
            # print(layeridx[i])
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    nc = np.load(
        "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    sig_mask = nc >= 10
    layeridx[~sig_mask] = np.nan

    # # projecting value back to 3D space
    all_vals = project_vals_to_3d(layeridx, cortical_mask)

    layerwise_volume = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=layer_num,
    )
    return layerwise_volume


def make_volume(
    subj,
    model=None,
    vals=None,
    model2=None,
    mask_with_significance=False,
    measure="corr",
    noise_corrected=False,
    cmap="hot",
    fdr_mask_name=None,
    vmin=0,
    vmax=None,
    roi="",
    n_session=34
):
    if vmax is None:
        if measure == "rsq":
            vmax = 0.6
        else:
            vmax = 1
        if model2 is not None:
            vmax -= 0.3
        if noise_corrected:
            vmax = 0.85
        if measure == "pvalue":
            vmax = 0.06

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    # notice
    tag = ""
    if roi != "":
        tag += "_%s" % roi
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d%s.npy"
        % (OUTPUT_ROOT, subj, subj, tag)
    )
    # nc = np.load(
    #     "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
    #     % (OUTPUT_ROOT, subj, subj)
    # )

    # load correlation scores of cortical voxels
    if vals is None:
        if (
            type(model) == list
        ):  # for different naming convention for variance partitioning (only 1 should exist)
            model_list = model
            for model in model_list:
                try:
                    vals = load_model_performance(
                        model, output_root=OUTPUT_ROOT, subj=subj, measure=measure
                    )
                    break
                except FileNotFoundError:
                    continue
        else:
            vals = load_model_performance(
                model, output_root=OUTPUT_ROOT, subj=subj, measure=measure, 
                roi = roi, n_session=n_session
            )
            # notice
        print("model:" + model)

        if model2 is not None:  # for variance paritioning
            vals2 = load_model_performance(
                model2, output_root=OUTPUT_ROOT, subj=subj, measure=measure
            )
            vals = vals - vals2
            print("model2:" + model2)
    # print(vals.shape)
    # print(np.mean(vals), np.min(vals), np.max(vals))
    if mask_with_significance:
        if args.sig_method == "fdr":
            sig_mask = load_fdr_mask(OUTPUT_ROOT, model, fdr_mask_name, subj, roi, n_session)
            print("sig_mask")
            print(type(sig_mask))
            print(sig_mask)
            if sig_mask is None:
                pass
                # print("Masking vals with nc only")
                # sig_mask = nc >= 10

    #     elif args.sig_method == "pvalue":
    #         pvalues = load_model_performance(
    #             model, output_root=OUTPUT_ROOT, subj=subj, measure="pvalue"
    #         )
    #         sig_mask = pvalues <= 0.05
        print(
            "Mask name: "
            + str(fdr_mask_name)
            + ". # of sig voxels: "
            + str(np.sum(sig_mask))
        )
        print("vals:",vals.shape)
        try:
            vals[~sig_mask] = np.nan
        except IndexError:
            non_zero_mask = np.load(
                "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
                % (OUTPUT_ROOT, subj, subj)
            )
            print("Masking zero voxels with mask...")
            sig_mask_tmp = np.zeros(non_zero_mask.shape)
            sig_mask_tmp[non_zero_mask] = sig_mask
            sig_mask = sig_mask_tmp.astype(bool)
            vals[~sig_mask] = np.nan
        
#notice
    # if (measure == "rsq") and (noise_corrected) and False:
    #     vals = vals / (nc / 100)
    #     vals[np.isnan(vals)] = np.nan
    # print("max:" + str(max(vals[~np.isnan(vals)])))

    # projecting value back to 3D space

    zeros_mask = (vals<=0)
    print("zeros_size", np.sum(zeros_mask==True))
    # vals[zeros_mask] = np.nan
    all_vals = project_vals_to_3d(vals, cortical_mask)
    vol_data = cortex.dataset.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def make_pc_volume(subj, vals, vmin=-0.5, vmax=0.5, cmap="BrBG"):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.dataset.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def vis_roi_ind():
    roi_mask = np.load(
        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-bodies.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    try:  # take out zero voxels
        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        roi_mask = roi_mask[non_zero_mask]
    except FileNotFoundError:
        pass

    mask = roi_mask == 1

    print(str(sum(mask)) + " voxels for optimization")
    vindx = np.arange(sum(mask))
    vals = np.zeros(roi_mask.shape)
    vals[mask] = vindx
    new_vals = np.zeros(non_zero_mask.shape)
    new_vals[non_zero_mask] = vals
    return new_vals


def make_3pc_volume(subj, PCs):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    pc_3d = []
    for i in range(3):
        tmp = PCs[i, :] / np.max(PCs[i, :]) * 255
        # projecting value back to 3D space
        pc_3d.append(project_vals_to_3d(tmp, cortical_mask))

    red = cortex.dataset.Volume(
        pc_3d[0].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    green = cortex.dataset.Volume(
        pc_3d[1].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    blue = cortex.dataset.Volume(
        pc_3d[2].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )

    vol_data = cortex.dataset.VolumeRGB(
        red,
        green,
        blue,
        "subj%02d" % subj,
        channel1color=(194, 30, 86),
        channel2color=(50, 205, 50),
        channel3color=(30, 144, 255),
    )

    return vol_data


def make_roi_volume(roi_name):
    roi = nib.load("%s/%s.nii.gz" % (ROI_FILE_ROOT, roi_name))
    roi_data = roi.get_fdata()
    roi_data = np.swapaxes(roi_data, 0, 2)

    roi_volume = cortex.dataset.Volume(
        roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=np.max(roi_data),
    )
    return roi_volume


def show_voxel_diff_in_repr_samples(model1, model2, quad="br"):
    def load_brain_response(model1, model2, quad):
        import glob

        fname = glob.glob(
            "./output/rdm_based_analysis/subj%d/voxel_corr_%s_vs_%s_*%s.npy"
            % (args.subj, model1, model2, quad)
        )
        vals = np.load(fname[0])

        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        tmp = np.zeros(non_zero_mask.shape)
        tmp[non_zero_mask] = vals
        vals = tmp

        all_vals = project_vals_to_3d(vals, cortical_mask)
        return all_vals

    if quad == "br-tl":
        b1 = load_brain_response(model1, model2, "br")
        b2 = load_brain_response(model1, model2, "tl")
        all_vals = b1 - b2
    else:
        all_vals = load_brain_response(model1, model2, quad)

    rdm_volume = cortex.Volume(
        all_vals,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=-0.1,
        vmax=0.1,
    )
    return rdm_volume


if __name__ == "__main__":
    import cortex
    import nibabel as nib

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--mask_sig", default=False, action="store_true")
    parser.add_argument("--sig_method", default="negtail_fdr")
    parser.add_argument("--alpha", default=0.05)
    parser.add_argument("--show_pcs", default=False, action="store_true")
    parser.add_argument("--show_clustering", default=False, action="store_true")
    parser.add_argument("--show_more", action="store_true")
    parser.add_argument("--show_repr_sim", action="store_true")
    parser.add_argument("--vis_method", type=str, default="webgl")
    parser.add_argument("--roi", type=str, default="")
    parser.add_argument("--session", type=int, default=34)
    # parser.add_argument("--root", type=str, default=".")

    args = parser.parse_args()
    print(args)

    import configparser

    config = configparser.ConfigParser()
    config.read("config.cfg")
    PPdataPath = config["DATA"]["PPdataPath"]
    ROI_FILE_ROOT = ("%s/subj%02d/func1pt8mm/roi" % (PPdataPath, args.subj))
    OUTPUT_ROOT = ROOT + "/result"

    roi = args.roi
    n_session = args.session

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d_%s.npy"
        % (OUTPUT_ROOT, args.subj, args.subj, roi)
    )
    volumes = {}

    volumes["clip-ViT-last R^2"] = make_volume(
        subj=args.subj,
        model="clip_vit",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
        roi = roi,
        n_session = n_session
    )

    volumes["clip-RN50-last R^2"] = make_volume(
        subj=args.subj,
        model="clip_visual_resnet",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        roi=roi,
        n_session = n_session
    )

    volumes["resnet50 R^2"] = make_volume(
        subj=args.subj,
        model="convnet_resnet",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        roi = roi,
        n_session = n_session
    )

    volumes["alex-last R^2"] = make_volume(
        subj = args.subj,
        model = "convnet_alexnet",
        mask_with_significance = args.mask_sig,
        measure="rsq",
        roi = roi,
        n_session = n_session
    )

    if args.show_pcs:
        model = "clip"
        # name_modifiers = ["best_20000_nc", "floc-bodies_floc-places_floc-faces_only", "floc-bodies_only", "floc-faces_only", "floc-places_only", "EBA_only", "OPA_only"]
        name_modifiers = ["best_20000_nc"]
        for name_modifier in name_modifiers:
            # visualize PC projections
            subj_proj = np.load(
                "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
                % (OUTPUT_ROOT, model, name_modifier, args.subj)
            )
            subj_mask = np.load(
                "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
                % (OUTPUT_ROOT, model, name_modifier, args.subj)
            )
            # proj_val_only = subj_proj[]

            # proj_vals = np.zeros(subj_proj.shape)
            # proj_vals[:, ~subj_mask] = np.nan
            # proj_vals[:, subj_mask] = subj_proj
            subj_proj_nan_out = subj_proj.copy()
            subj_proj_nan_out[:, ~subj_mask] = np.nan

            for i in range(subj_proj.shape[0]):
                key = "Proj " + str(i) + name_modifier
                volumes[key] = make_pc_volume(
                    args.subj,
                    subj_proj_nan_out[i, :],
                )

            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(np.sum(subj_proj ** 2, axis=1))
            plt.savefig("figures/PCA/proj_norm_%s.png" % name_modifier)

            plt.figure()
            plt.hist(subj_proj[0, :], label="0", alpha=0.3)
            plt.hist(subj_proj[1, :], label="1", alpha=0.3)
            plt.legend()
            plt.savefig("figures/PCA/proj_hist_%s.png" % name_modifier)

        # volumes["3PC"] = make_3pc_volume(
        #     args.subj,
        #     PCs_zscore,
        # )

        # # basis?
        # def kmean_sweep_on_PC(n_pc):
        #     from sklearn.cluster import KMeans

        #     inertia = []
        #     for k in range(3, 10):
        #         kmeans = KMeans(n_clusters=k, random_state=0).fit(
        #             subj_proj[:n_pc, :].T
        #         )
        #         volumes["basis %d-%d" % (n_pc, k)] = make_pc_volume(
        #             args.subj, kmeans.labels_, vmin=0, vmax=k-1, cmap="J4s"
        #         )
        #         inertia.append(kmeans.inertia_)
        #     return inertia

        # import matplotlib.pyplot as plt

        # plt.figure()
        # n_pcs = [3, 5, 10]
        # for n in n_pcs:
        #     inertia = kmean_sweep_on_PC(n)
        #     plt.plot(inertia, label="%d PCS" % n)
        # plt.savefig("figures/pca/clustering/inertia_across_pc_num.png")

        # # visualize PC projections
        # subj_proj = np.load(
        #             "%s/output/pca/%s/subj%02d/%s_feature_pca_projections.npy"
        #             % (OUTPUT_ROOT, model, args.subj, model)
        #         )
        # for i in range(PCs.shape[0]):
        #     key = "PC Proj " + str(i)
        #     volumes[key] = make_pc_volume(
        #         args.subj,
        #         subj_proj[i, :],
        #     )

        # mni_data = project_vols_to_mni(s, volume)

        # mni_vol = cortex.Volume(
        #     mni_data,
        #     "fsaverage",
        #     "atlas",
        #     cmap="inferno",
        # )

        # cortex.quickflat.make_figure(mni_vol, with_roi=False)
        # print("***********")
        # print(volumes["PC1"])

    if args.show_clustering:
        model = "clip"

        # name_modifier = "acc_0.3_minus_prf-visualrois"
        name_modifier = "best_20000_nc"
        labels_vals = np.load(
            "%s/output/clustering/spectral_subj%01d.npy" % (OUTPUT_ROOT, args.subj)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
            % (OUTPUT_ROOT, model, name_modifier, args.subj)
        )

        labels = np.zeros(subj_mask.shape)
        labels[~subj_mask] = np.nan
        labels[subj_mask] = labels_vals

        # volumes["spectral clustering"] = make_pc_volume(
        #             args.subj, labels, vmin=0, vmax=max(labels), cmap="J4s"
        #         )

    if args.vis_method == "webgl":
        subj_port = "4111" + str(args.subj)
        # cortex.webgl.show(data=volumes, autoclose=False, port=int(subj_port))
        cortex.webgl.show(data=volumes, port=int(subj_port), recache=False)

    elif args.vis_method == "MNI":
        group_mni_data = []
        mni_mask = cortex.utils.get_cortical_mask("fsaverage", "atlas")

        for s in range(1, 9):
            print("SUBJECT: " + str(s))

            # visualize in MNI space

            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (OUTPUT_ROOT, s, s)
            )
            mask = cortex.utils.get_cortical_mask(
                "subj%02d" % s, "func1pt8_to_anat0pt8_autoFSbbr"
            )

            volume = make_volume(
                subj=s,
                model="YFCC_slip_YFCC_simclr",
                model2="YFCC_simclr",
                mask_with_significance=args.mask_sig,
                measure="rsq",
                cmap="inferno",
                # fdr_mask_name=["YFCC_simclr-YFCC_slip_unique_var", "YFCC_slip-YFCC_simclr_unique_var"]
            )
            mni_data = project_vols_to_mni(s, volume)
            # group_mni_data.append(mni_data.flatten())
            group_mni_data.append(mni_data[mni_mask])

            # saving MNI flatmap for individual subjects
            volume_masked = make_volume(
                subj=s,
                model="YFCC_slip_YFCC_simclr",
                model2="YFCC_simclr",
                mask_with_significance=args.mask_sig,
                measure="rsq",
                cmap="inferno",
                fdr_mask_name=[
                    "YFCC_simclr-YFCC_slip_unique_var",
                    "YFCC_slip-YFCC_simclr_unique_var",
                ],
            )
            mni_data_masked = project_vols_to_mni(s, volume_masked)
            # mni_3d = np.zeros(mni_mask.shape)
            # mni_3d[mni_mask] = mni_data_masked
            subj_vol = cortex.Volume(
                mni_data_masked,
                "fsaverage",
                "atlas",
                cmap="inferno",
                vmin=0,
                vmax=0.05,
            )
            filename = "figures/flatmap/mni/subj%d_YFCC_slip_unique_var_sig_vox.png" % s
            _ = cortex.quickflat.make_png(
                filename,
                subj_vol,
                linewidth=1,
                labelsize="17pt",
                with_curvature=True,
                recache=False,
                # roi_list=roi_list,
            )

        from scipy.stats import ttest_1samp
        from statsmodels.stats.multitest import fdrcorrection

        t, p_val = ttest_1samp(np.array(group_mni_data), 0, alternative="greater")
        print(t)
        print(p_val.shape)
        # p_val = p_val.reshape(mni_data.shape)

        # np.save("%s/output/ci_threshold/YFCC_slip_group_t_test_p.npy" % OUTPUT_ROOT, p_val)
        np.save("%s/output/ci_threshold/YFCC_slip_group_t_test_t.npy" % OUTPUT_ROOT, t)
        # import pdb

        # pdb.set_trace()

        p_val_to_show = p_val.copy()
        p_val_to_show[p_val > 0.05] = np.nan

        p_val_3d = np.zeros(mni_mask.shape)
        p_val_3d[mni_mask] = p_val_to_show
        # all_vals = np.swapaxes(all_vals, 0, 2)

        # p_val_to_show_3d = project_vals_to_3d(p_val_to_show, mni_mask)
        p_vol = cortex.Volume(
            p_val_3d,
            "fsaverage",
            "atlas",
            cmap="Blues_r",
            vmin=0,
            vmax=0.05,
        )

        # -log10

        # cortex.quickflat.make_figure(p_vol, with_roi=False)
        filename = "figures/flatmap/mni/group_YFCC_slip_p.png"
        _ = cortex.quickflat.make_png(
            filename,
            p_vol,
            linewidth=1,
            labelsize="17pt",
            with_curvature=True,
            recache=False,
            # roi_list=roi_list,
        )

        mask_for_nan = ~np.isnan(p_val)
        fdr_p = fdrcorrection(p_val[mask_for_nan])[1]
        # fdr_p = fdrcorrection(p_val)[1]

        fdr_p_full = np.ones(p_val.shape)
        fdr_p_full[mask_for_nan] = fdr_p

        fdr_p_val_3d = np.zeros(mni_mask.shape)
        fdr_p_val_3d[mni_mask] = fdr_p_full

        fdr_p_vol = cortex.Volume(
            fdr_p_val_3d,
            "fsaverage",
            "atlas",
            cmap="Blues_r",
            vmin=0,
            vmax=0.05,
        )

        # -log10

        # cortex.quickflat.make_figure(p_vol, with_roi=False)
        filename = "figures/flatmap/mni/group_YFCC_slip_fdr_p.png"
        _ = cortex.quickflat.make_png(
            filename,
            fdr_p_vol,
            linewidth=1,
            labelsize="17pt",
            with_curvature=True,
            recache=False,
            # roi_list=roi_list,
        )

        np.save(
            "%s/output/ci_threshold/YFCC_slip_group_t_test_fdr_p.npy" % OUTPUT_ROOT,
            fdr_p_full,
        )

    elif args.vis_method == "quickflat":
        roi_list = ["RSC", "PPA", "OPA", "EBA", "EarlyVis", "FFA-1", "FFA-2"]
        for k in volumes.keys():
            # vol_name = k.replace(" ", "_")
            # notice dir
            root = "%s/figures/flatmap/subj%d" % (OUTPUT_ROOT, args.subj)
            if not os.path.exists(root):
                os.makedirs(root)
            print(volumes[k])
            filename = "%s/%s_%s_%d_session.png" % (root, k, roi, n_session)
            _ = cortex.quickflat.make_png(
                filename,
                volumes[k],
                linewidth=3,
                labelsize="20pt",
                with_curvature=True,
                recache=False,
                roi_list=roi_list,
            )