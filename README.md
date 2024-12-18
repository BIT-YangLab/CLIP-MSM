# CLIP-MSM

> A Multi-Semantic Mapping Brain Representation for Human High-Level Visual Cortex

## Setup

```bash
git clone https://github.com/mofianger/clip-msm.git
cd clip-msm
```

## Install Dependencies

Ensure you have the necessary packages by referring to the requirements file. Set up a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install `torch` and `torchvision` from [PyTorch](https://pytorch.org/) tailored to your system configuration.

## Data acquisition

To proceed with the analysis, you'll need specific datasets. For a streamlined setup, consider downloading the required datasets directly from their respective sources, such as the [natural scene datasets](https://naturalscenesdataset.org/) and the NOD

## Model Training

This part is located within clip-msm/model-training directory

### Configure Paths

Examine the sample `config.cfg` for guidance on setting up paths. Update it to include the local paths for your NSD data, NOD data, coco, etc.

The directory structure of result should be organized as follows where `ROOT` shall be your working directory:

```
ROOT/result/
├── output/          # For output files like encoding models and bootstrap results.
├── figures/         # For diagrams and images.
└── features/        # For files associated with model features.
```

### One-command run

`project_commands.sh` is designed to run all models for all subjects by default. 

To execute the script:

```bash
sh project_commands.sh
```

### Run single models

#### On NSD

To run a specific model on the NSD, utilize the command:

```bash
python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_vit
python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_visual_resnet
python run_model/run_alex.py --subj $subj
python run_model/run_resnet.py --subj $subj
```

#### On NOD

For the NOD dataset, run the model using:

```bash
python run_model/RN50_NOD.py --subj $subj
python run_model/ViT_NOD.py --subj $subj
```

### Bootstrap

To process bootstrap result, you can use:

``` bash
python analyze/process_bootstrap_results.py --subj $subj --model clip_vit --roi SELECTIVE_ROI
```

### Visualization

For visualizing results, execute the following scripts located in the `/plot` directory:

- **Noise Ceiling**: run `plot_noise_ceiling.py`
- **Model Performance**:
  - `plot_box_graph.py` generates a box plot illustrating the performance of all models on a specified subject.
  - `plot_multiple_models.py` displays the mean RSQ (R-squared) and its standard deviation for all models across various data sessions on a specific subject.
  - `plot_single_model.py` displays the mean RSQ and its standard deviation for a single model across different data sessions on a specific subject.
  - `plot_single_model_Pearson.py` assesses model performance using Pearson correlation coefficients.
- **Flatmap Visualization**: Utilize the script `plot_flatmap.sh`.

The visualizing of CLIP-MSM is based on the results of Model Training and Dissect. 
 - `visualization_3d_view.py` is used to visualize any result on the surface.
 - `visualize_FOOD_ROI.py` and `visualize_in_pycortex_FOOD_ROI.py` are used to visualize the food selevtive regions on the native space flatmap.
 - `visualize_resnet50_training.py` and `visualize_in_pycortex_2.py` are used to visualize the performance of all encoding models on the native space flatmap.
 - `visualize_hard-clip-dissect.py` and `visualize_in_pycortex_hard_clip_dissect.py` are used to visualize voxel-wise semantic mapping on the native space flatmap.
 - `visualize_hard-clip-dissect-label.py` and `visualize_in_pycortex_hard_clip_dissect_label.py` are used to visualize semantic mapping of individual label on the native space flatmap.
 - `visualize_soft-clip-dissect.py` and `visualize_in_pycortex_soft_clip_dissect.py` are used to visualize the multi-semantic-mapping of all voxels on the native space flatmap.
 - `visualize_SELECTIVE_ROI.py` and `visualize_in_pycortex_SELECTIVE_ROI.py` are used to visualize the Selevtive ROI on the native space flatmap.

## Dissect

### Config Settings

You need to replace the dataset path in the config file with your local path before running the dissection model. Also, you can choose your running device, cpu or cuda

The directory structure of result will be organized as follows where `ROOT` shall be your working directory:

```
ROOT/result/
├── NOD_DissectActivation/          # Preserve the model's embedding of the image on NOD dataset
├── NSD_DissectActivation/         # Preserve the model's embedding of the image on NSD dataset
└── NOD_DissectResult/        # Preservation of voxel dissection results and visualization of the highest voxel activation images
└── NSD_DissectResult/        # Preserve the result of NSD Dataset
```
### One-command run
`run.sh` is designed to run all models for all subjects in NOD dataset by default.
```bash
sh run.sh
```
`NOD_run.sh` is designed to run all models for all subjects in NSD dataset by default.

```bash
sh NOD_run.sh
```
### Run for single model

You can choose your target model and subject by running `describe_neurons.py` for NSD dataset and `NOD_describe_neurons.py` for NOD dataset

## Plot

### `voxal_top_image.py`

You can use this function to get multiple images of the maximally activated voxel,

### `barfigDrawer.py`

You can use this file to get a histogram of profiling results for specific brain regions

### `ResultAnalysis/NSD_main.py or NOD_main.py`

You can use this file to calculate the correlation between the soft and hard dissection results and the floc response in NSD or NOD dataset dissection result.

### `ResultAnalysis/NSD_boxplot_test.py or NOD_boxplot_test.py`

Use this file to compute the t test value between the soft and hard dissection method, and plot the box-and-line graphs for NSD or NOD dataset dissection result.

# 
This streamlined guide should help you set up and execute the CLIP-MSM project efficiently.
