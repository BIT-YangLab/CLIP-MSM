python src/initial.py

# for NSD subj 1-8

MODELS="
clip_vit \
clip_visual_resnet \
convnet_resnet \
convent_alexnet"

for subj in $(seq 1 8); do
    echo "processing subj $subj"
    # extract trial ID list
    python src/extract_image_list.py --subj $subj --type trial
    python src/extract_image_list.py --subj $subj --type cocoId

    # prepare brain voxels for encoding models:
    #   - extract cortical mask;
    #   - mask volume metric data;
    #   - zscore data by runs

    python src/extract_cortical_voxel.py --zscore_by_run --subj $subj --roi SELECTIVE_ROI


    # computer explainable variance for the data and output data averaged by repeats

    python src/compute_ev.py --subj $subj --zscored_input --compute_ev --roi SELECTIVE_ROI

    # extract model features
    python src/extract_clip_features.py --subj $subj
    python src/extract_alexnet_features.py --subj $subj
    python src/extract_convnet_features.py --subj $subj

    # run encoding models
    python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_vit
    python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_visual_resnet
    python run_model/run_alex.py --subj $subj
    python run_model/run_resnet.py --subj $subj

    # processing bootstrap test results
    python analyze/process_bootstrap_results.py --subj $subj --model clip_vit --roi SELECTIVE_ROI
    python analyze/process_bootstrap_results.py --subj $subj --model clip_visual_resnet --roi SELECTIVE_ROI
    python analyze/process_bootstrap_results.py --subj $subj --model convnet_resnet --roi SELECTIVE_ROI
    python analyze/process_bootstrap_results.py --subj $subj --model convnet_alexnet --roi SELECTIVE_ROI

    # visualizing results
    python src/visualize_in_pycortex.py --subj $subj --mask_sig --sig_method fdr --vis_method quickflat --roi SELECTIVE_ROI

done

# for NOD subj 1-9

# extract NOD brain data
python src/extract_NOD.py

# make mask
python src/make_floc_mask.py

# extract image feature
python src/describe_neurons-feature.py

for subj in $(seq 1 9); do
    echo "processing subj $subj"
    python run_model/RN50_NOD.py --subj $subj
    python run_model/ViT_NOD.py --subj $subj

done