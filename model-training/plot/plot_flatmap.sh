for subj in $(seq 1 1); do
    python src/visualize_in_pycortex.py --subj $subj --sig_method fdr --vis_method quickflat --roi SELECTIVE_ROI --session 34
done