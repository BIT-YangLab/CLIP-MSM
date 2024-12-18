import numpy as np
import argparse

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"]

def fdr_correct_p(var):
    from statsmodels.stats.multitest import fdrcorrection

    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    print(p_vals[:20])
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p

def process_bootstrap(subj, model, roi, root, session=[34]):
    for s in session:
        rsq = np.load( "%s/output/bootstrap/%s_%s/subj%01d/%d_session/rsq_dist_%s_%s.npy"
            % (root, model, roi, subj, s, model, roi))
        
        fdr_p = fdr_correct_p(rsq)

        # import mystat
        # mystat.stat(fdr_p[1], name="fdr_p.png")

        import os
        os.makedirs("%s/output/ci_threshold" % (root), exist_ok=True)
        np.save(
            "%s/output/ci_threshold/%s_%s_%d_session_fdr_p_subj%01d.npy"
            % (root, model, roi, s, subj),
            fdr_p,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="please specific subject")
    parser.add_argument(
        "--subj", type=int, default=5, help="specify which subject to process bootstrap"
    )
    parser.add_argument("--roi", type=str, default="")
    parser.add_argument("--model", type=str, default="clip_visual_resnet")
    # parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = ROOT + "/result"

    output_dir = root + "/output"
    figure_dir = root + "/figures"
    features_dir = root + "/features"

    subj = args.subj
    roi = args.roi
    model = args.model

    session = [5, 10, 15, 20, 25, 30, 34]
    process_bootstrap(subj, model, roi, root, session)