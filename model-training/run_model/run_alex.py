# model = "convnet_alexnet"
fix_testing = 42 # random seed
# roi = "SELECTIVE_ROI"

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

from scipy.stats import pearsonr

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
    # a data session contain 10000/40 = 250 pieces of photos
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

            path = "%s/bootstrap/%s/subj%d/%d_session" % (saving_dir, model_name,subj, n)
            if not os.path.isdir(path):
                os.makedirs(path)
            np.save("%s/rsq_dist_%s.npy" % (path, model_name), rsq_dists)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--roi", type=str, default="SELECTIVE_ROI")
    parser.add_argument("--model", type=str, default="convnet_alexnet")
    # parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    output_dir = ROOT + "/result/output"
    figure_dir = ROOT + "/result/figures"
    features_dir = ROOT + "/result/features"

    subj = args.subj
    roi = args.roi
    model = args.model

    # Load brain data
    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d%s.npy"
        % (output_dir, subj, roi)
    )

    br_data = np.load(brain_path)
    print(br_data.shape)

    trial_mask = np.sum(np.isnan(br_data), axis=1) <= 0
    br_data = br_data[trial_mask, :]

    print("NaNs? Finite?:")
    print(np.any(np.isnan(br_data)))
    print(np.all(np.isfinite(br_data)))
    print("Brain response size is: " + str(br_data.shape))

    stimulus_list = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (output_dir, subj)
    )

    # load feature matrix

    layer_modifier = "_avgpool"
    featmat = np.load(
                "%s/subj%d/%s%s.npy" % (features_dir, subj, model, layer_modifier)
            )
      
    feature_mat = featmat.squeeze()
    feature_mat = feature_mat[trial_mask, :]

    fm = feature_mat
    br = br_data

    # whether set cross-validation True or False

    cv = False
    
    print("fix_testing:", fix_testing)

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