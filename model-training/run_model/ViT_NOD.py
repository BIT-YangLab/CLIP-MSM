model = "ViT_NOD"

fix_testing = 42 # random seed

import pickle
import numpy as np
import os
import argparse
from sklearn.model_selection import (
    KFold,
    PredefinedSplit,
    train_test_split,
    ShuffleSplit,
)
import torch
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


from encodingmodel.ridge import RidgeCVEstimator

from scipy.stats import pearsonr, zscore

from util.util import r2_score

from tqdm import tqdm

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"]

def scoring(y, yhat):
    return -torch.nn.functional.mse_loss(yhat, y)

def bootstrap_sampling(weights, bias, X_mean, X_test, y_test, repeat, seed):
    np.random.seed(seed)
    rsq_dist = list()
    label_idx = np.arange(X_test.shape[0])
    # yhat = (X_test - X_mean) @ weights + bias
    for _ in tqdm(range(repeat)):
        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
        X_test_sampled = X_test[sampled_idx, :]
        yhat = (X_test_sampled - X_mean) @ weights + bias
        y_test_sampled = y_test[sampled_idx, :]
        rsqs = r2_score(y_test_sampled, yhat.cpu().numpy())
        rsq_dist.append(rsqs)

    return rsq_dist

def fit_encoding_model(
    X,
    y,
    model_name=None,
    subj=1,
    fix_testing=None,
    cv=False,
    saving=True,
    saving_dir=None,
    divided=40,
    tested=34,
    roi=""
):

    if roi == "":
        model_name += "_whole_brain"
    else:
        model_name += "_%s" % roi

    if cv:
        print("Running cross validation")

    out_path = "%s/encoding_results/%s/subj%d" % (saving_dir, model_name, subj)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    assert (
        y.shape[0] == X.shape[0]
    )  # test that shape of features spaces and the brain are the same

    # 按固定顺序打乱

    order_li = np.arange(X.shape[0])
    np.random.seed(fix_testing)
    np.random.shuffle(order_li)
    print(order_li[:10])

    X_shuffled = X[order_li]
    y_shuffled = y[order_li]

    piece = X.shape[0] // divided

    num = tested * piece

    X_test, y_test = X_shuffled[num:], y_shuffled[num:]
    
    X_test = torch.from_numpy(X_test).to(dtype=torch.float64).to(device)

    tol = 8

    alphas = torch.from_numpy(
            np.logspace(-tol, 1 / 2 * np.log10(X.shape[1]) + tol, 100)
        )
    # alpha 正则化 选择最佳的参数

    n_session = [5, 10, 15, 20, 25, 30, 34]
    # n_session = [5]

    for n in n_session:
        outpath = "%s/%d_session" % (out_path, n)
        check_path = "%s/weights_%s.npy" % (outpath, model_name)
        data_num = n * piece
        X_train = X_shuffled[:data_num]
        y_train = y_shuffled[:data_num]

        X_train = torch.from_numpy(X_train).to(dtype=torch.float64).to(device)
        y_train = torch.from_numpy(y_train).to(dtype=torch.float64).to(device)

        if False and os.path.exists(check_path):
            print("already exist! skipped fitting model")
        else:
            print(check_path)
            print("no existed yet") 

            if cv:
                nfold = 7
                kfold = KFold(n_splits=nfold)
            else:
                tr_index, _ = next(
                    ShuffleSplit(test_size=0.15).split(
                        X_train, y_train
                    )  # split training and testing
                )
                # set predefined train and validation split
                test_fold = np.zeros(X_train.shape[0])
                test_fold[tr_index] = -1
                kfold = PredefinedSplit(test_fold)
                assert kfold.get_n_splits() == 1
            
            clf = RidgeCVEstimator(alphas, kfold, scoring, scale_X=False)

            print("Fitting ridge models...")

            clf.fit(X_train, y_train)

            weights, bias = clf.get_model_weights_and_bias()

            print("Making predictions using ridge models...")
            yhat = clf.predict(X_test).cpu().numpy()
            try:
                rsqs = r2_score(y_test, yhat)
            except ValueError:  # debugging for NaNs in subj 5
                print("Ytest: NaNs? Finite?")
                print(np.any(np.isnan(y_test)))
                print(np.all(np.isfinite(y_test)))
                print("Yhat: NaNs? Finite?")
                print(np.any(np.isnan(yhat)))
                print(np.all(np.isfinite(yhat)))

            corrs = [pearsonr(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]

            cv_outputs =  (
                corrs,
                rsqs,
                clf.mean_cv_scores.cpu().numpy(),
                clf.best_l_scores.cpu().numpy(),
                clf.best_l_idxs.cpu().numpy(),
                [yhat, y_test],
                weights.cpu().numpy(),
                bias.cpu().numpy(),
                clf,
            )
            
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if saving:
                pickle.dump(cv_outputs[0], open(outpath + "/corr_%s.p" % model_name, "wb"))

                if len(cv_outputs) > 0:
                    pickle.dump(cv_outputs[1], open(outpath + "/rsq_%s.p" % model_name, "wb"))
                    pickle.dump(
                        cv_outputs[2],
                        open(outpath + "/cv_score_%s.p" % model_name, "wb"),
                    )
                    pickle.dump(
                        cv_outputs[3],
                        open(outpath + "/l_score_%s.p" % model_name, "wb"),
                    )
                    pickle.dump(
                        cv_outputs[4],
                        open(outpath + "/best_l_%s.p" % model_name, "wb"),
                    )

                    if fix_testing:
                        pickle.dump(
                            cv_outputs[5],
                            open(outpath + "/pred_%s.p" % model_name, "wb"),
                        )

                    np.save("%s/weights_%s.npy" % (outpath, model_name), cv_outputs[6])
                    np.save("%s/bias_%s.npy" % (outpath, model_name), cv_outputs[7])
                    pickle.dump(
                        cv_outputs[8], open("%s/clf_%s.pkl" % (outpath, model_name), "wb")
                    )
        # running bootstrap
        path = "%s/bootstrap/%s/subj%d/%d_session" % (saving_dir, model_name,subj, n)
        check_path = "%s/rsq_dist_%s.npy" % (path, model_name)
        if False and os.path.exists(check_path):
            print("already exist! skipped bootstrap")
        else:
            weights = np.load(
                "%s/weights_%s.npy"
                % (outpath, model_name)
            )
            bias = np.load(
                "%s/bias_%s.npy"
                % (outpath, model_name)
            )
            weights = torch.from_numpy(weights).to(dtype=torch.float64).to(device)
            bias = torch.from_numpy(bias).to(dtype=torch.float64).to(device)

            X_mean = X_train.mean(dim=0, keepdim=True)

            rsq_dists = bootstrap_sampling(
                weights, bias, X_mean, X_test, y_test, repeat=2000, seed=41
            )

            if not os.path.isdir(path):
                os.makedirs(path)
            np.save("%s/rsq_dist_%s.npy" % (path, model_name), rsq_dists)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--roi", type=str, default="")
    parser.add_argument("--model", type=str, default="ViT_NOD")
    # parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    output_dir = ROOT + "/result/output"

    subj = args.subj
    roi = args.roi
    model = args.model

    # Load brain data
    brain_ori=np.load(ROOT + '/result/features/NOD/sub-%02d_imagenet-beta_hp128_s4_ridge.npy'%(subj))
    brain_coco=np.load(ROOT + '/result/features/NOD/sub-%02d_coco-beta_hp128_s4_ridge.npy'%(subj))
    
    import nibabel as nib
    mask=nib.load(ROOT +'/result/mask/floc_roi_sub%d.dlabel.nii' % subj)
    mask=mask.get_fdata()
    mask=np.array(mask)
    mask = mask.squeeze()>0

    brain_data = brain_ori[:,:,mask]
    brain_masked_coco=brain_coco[:,:,mask]

    brain_z_coco=zscore(brain_masked_coco,axis=1)
    brain_res_coco=np.mean(brain_z_coco,axis=0)

    brain_res=np.concatenate((brain_data[0],brain_data[1],brain_data[2],brain_data[3]))

    def zscore_by_run(mat, run_data = 100):
        from scipy.stats import zscore
        run_n = mat.shape[0] / run_data
        zscored_mat = np.zeros(mat.shape)
        index_so_far = 0
        for i in tqdm(range(int(run_n))):
            zscored_mat[index_so_far : index_so_far + run_data, :] = zscore(
                mat[index_so_far : index_so_far + run_data, :]
            )
            index_so_far += run_data
        return zscored_mat
    
    brain_res_z = zscore_by_run(brain_res)
    br_data = np.zeros((4120,brain_res.shape[1]))
    print(brain_res_z.shape, brain_res_coco.shape)
    br_data = np.concatenate((brain_res_z, brain_res_coco), axis = 0)

    print("Brain response size is: " + str(br_data.shape))

    # load feature matrix

    # this is for CLIP_ViT

    featmat_coco=np.array(torch.load(ROOT + '/result/features/NOD/nod_coco_ViT-B_32_last_layer.pt'))
    featmat=torch.load(ROOT + '/result/features/NOD/nod_sub%02d_ViT-B_32_last_layer.pt' % (subj))
    featmat=np.array(featmat)
    featmat=np.concatenate((featmat,featmat_coco))


    print("Feature size is: " + str(featmat.shape))

    fm = featmat
    br = br_data

    # whether set cross-validation True or False

    cv = False
    
    fit_encoding_model(
            fm,
            br,
            model,
            subj,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            saving_dir=output_dir,
            divided = 40,
            roi = roi
        )