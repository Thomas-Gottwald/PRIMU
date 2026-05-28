# PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations

| [Project Page](https://osvia.org/PRIMU/) | [Paper](https://arxiv.org/abs/2508.02443) |

## Overview

This repository contains the official implementation of:

> **PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage**

The main implementation is based on the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) framework and is located in `./3dgs`.

This repository currently provides code for:

* Uncertainty Estimation (UE)
* Active View Selection (AVS)

Code for the scene separation study will be released soon.

---

## Setup

Setup instructions for PRIMU uncertainty estimation and active view selection:

```sh
conda create -n primu python=3.12
conda activate primu

pip install torch==2.9.1 torchvision==0.24.1

pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 --no-build-isolation
pip install PRIMU/3dgs/submodules/simple-knn --no-build-isolation
pip install PRIMU/3dgs/submodules/diff-gaussian-rasterization_bp --no-build-isolation
pip install PRIMU/3dgs/submodules/fused-ssim --no-build-isolation

pip install tqdm einops==0.8.1 s2fft plyfile opencv-python matplotlib scikit-learn pandas
```

<details>
<summary><span style="font-weight: bold;">Troubleshooting</span></summary>

* You may need to install a different GCC version. This can be done via conda-forge:

```sh
conda install cxx-compiler gcc=9.4 -c conda-forge
```

</details>

---

## Usage

### Uncertainty Estimation

### 1. Train Gaussian Splatting

```sh
python PRIMU/3dgs/train.py \
--source_path <input_path> \
--model_path <scene_path> \
--iterations 30000 \
--eval
```

<details>
<summary><span style="font-weight: bold;">Important parameters</span></summary>

* `-s`, `--source_path`
  Path to the source images of the scene.

* `-m`, `--model_path`
  Path where the Gaussian Splatting model will be stored.

* `--iterations`
  Number of training iterations.

* `--resolution`
  Downscaling factor for the input image resolution.

* `--eval`
  Holds out a subset of images for evaluation.

* `--useLF360`
  Uses the 360° training-view setup for the LF dataset.

</details>

---

### 2. Create uncertainty feature maps

Compute the uncertainty primitive representation and render the corresponding uncertainty feature maps.

```sh
python PRIMU/3dgs/PRIMU_create_feature_maps.py \
-m <scene_path> \
--sh_deg 4 \
--plot test
```

<details>
<summary><span style="font-weight: bold;">Important parameters</span></summary>

* `--sh_deg`
  Spherical Harmonics degree used for direction-dependent uncertainty feature maps.
  <br>**Default**: `0`

* `--kappa`
  Concentration parameter (`kappa`) of the von Mises–Fisher distribution used for direction-dependent feature maps.
  <br>**Default**: `8`

* `--store`
  List of dataset splits for which feature maps should be stored (`test`, `train`).
  <br>**Default**: `test`

* `--plot`
  List of dataset splits for which plots of the feature maps should be created.
  A split must also be included in `store` to enable plotting.

</details>

---

### 3. Train uncertainty regressors

Train regression models to predict pixel-wise RGB or depth errors and evaluate them.

```sh
python PRIMU/3dgs/PRIMU_regressor.py \
-s <scene_path> \
-i 0 1 2 3 \
--sh_deg 4 \
-x visibility_map_*_sh4 visibility_counter error_bp_map_*_sh4 \
--experiment_tag 1view \
--save_ue_maps \
--save_regressor \
--eval_on_train
```

<details>
<summary><span style="font-weight: bold;">Parameters</span></summary>

* `-e`, `--error_type`
  Type of error to predict (`rgb` or `depth`).
  <br>**Default**: `rgb`

* `-s`, `--scene_paths`
  Paths to Gaussian Splatting scene output directories.
  If multiple scene paths are provided, the regressor is trained jointly on all scenes.

* `-i`, `--holdout_indices`
  List of holdout-view indices or index combinations used for regressor training.
  Remaining holdout views are used for evaluation.

  For each entry, a separate regression model is trained.

  Examples:

  * `0-2` → range selection
  * `0+1+3` → custom selection

* `--sh_deg`
  Spherical Harmonics degree of the feature maps.
  <br>**Default**: `0`

* `--kappa`
  Concentration parameter (`kappa`) of the von Mises–Fisher distribution used for the feature maps.
  <br>**Default**: `8`

* `-x`, `--x_maps`
  List of feature maps used to train the regressor.
  The wildcard `*` is replaced by all entries from `map_variants`.
  <br>**Default**: `visibility_map_*_sh0 visibility_counter error_bp_map_*_sh0`

* `-v`, `--map_variants`
  List of map variants, e.g. different aggregation functions used for `x_maps`.
  <br>**Default**: `MAX MAXnoAlpha SUM SUMnoAlpha MEAN MEANnoAlpha`

* `-r`, `--regressor_model`
  Scikit-learn regression model:

  * `hgbr` → `HistGradientBoostingRegressor`
  * `lin` → `LinearRegression`
  <br>**Default**: `hgbr`

* `--experiment_tag`
  Optional tag used to organize experiments. Added to the output directory path.

* `--save_ue_maps`
  Stores the resulting uncertainty maps as `.npy` files.

* `--save_regressor`
  Saves the regression model as a pickle file.

* `--eval_on_train`
  Also evaluates on the training holdout views.
  Metrics are stored in a separate JSON file.

</details>

---

### 4. Calculate mean metrics

A helper script to compute mean metrics across different regressor training-view configurations.

```sh
python PRIMU/3dgs/PRIMU_calculate_metric_means.py \
-d <scene_path>/PRIMU/eval/<experiment_tag>/<error_type>_<regressor_model>_kappa<kappa:.2f>_sh<sh_deg>/test
```

<details>
<summary><span style="font-weight: bold;">Parameters</span></summary>

* `-d`, `--dir`
  Directory containing the JSON evaluation files.

* `-m`, `--metrics`
  List of uncertainty-estimation metrics for which mean values should be computed.
  <br>**Default**: `pearson spearman AUSE`

</details>

---

## Active View Selection

```sh
CUDA_VISIBLE_DEVICES=0 python PRIMU/3dgs/active_train.py \
--source_path <input_path> \
--model_path <scene_path> \
--schema v20seq1_inplace \
--method PRIMU \
--iterations 30000 \
--eval \
--advanced_report
```

<details>
<summary><span style="font-weight: bold;">Important parameters</span></summary>

General Gaussian Splatting parameters are identical to those of the standard training script.

* `--method`
  View-selection method (`PRIMU`, `rand`).
  <br>**Default**: `PRIMU`

* `--random_seed_for_view_selection`
  Uses a dedicated random seed for random view selection instead of `args.seed`.

* `--advanced_report`
  Stores additional active-training information such as candidate uncertainty masks.

* `--sh_deg`
  Spherical Harmonics degree of the feature maps.
  <br>**Default**: `4`

* `--kappa`
  Concentration parameter (`kappa`) of the von Mises–Fisher distribution used for direction-dependent feature maps.
  <br>**Default**: `8`

* `--feature_map`
  PRIMU feature map used for active view selection.
  <br>**Default**: `visibility_map_MAXnoAlpha_sh4`

* `--plot_selection`
  Creates visualization plots of the selected views, renderings, PRIMU feature maps, and camera positions.

</details>

---

## Data

| Dataset    | Download                                                               |
| ---------- | ---------------------------------------------------------------------- |
| MipNeRF360 | https://jonbarron.info/mipnerf360                                      |
| LLFF       | https://drive.google.com/file/d/11PhkBXZZNYTD2emdG1awALlhCnkq7aN-/view |
| LF         | https://drive.google.com/file/d/1U-Hly00DmqtAIGaPkF-Eu_B_q0Frsbh1/view |
| TUM        | https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download             |

Additional notes:

* **LLFF** alternative mirror:
  https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full?select=nerf_llff_data

* **TUM** dataset category:
  *3D Object Reconstruction*

### Reproducing the paper results

* **MipNeRF360**
  We use the same image resolutions as the original Gaussian Splatting paper:

  * Indoor scenes (`bonsai`, `counter`, `kitchen`, `room`): `--resolution 2`
  * Outdoor scenes (`bicycle`, `flowers`, `garden`, `stump`, `treehill`): `--resolution 4`

* **LLFF**
  We use automatic downscaling to a maximum image resolution of 1600 pixels.

* **LF**

  * Evaluation views correspond to views with available ground-truth depth.
  * The image resolution is matched to the ground-truth depth resolution.
  * For the base setup:

    * use the `--useLF360` flag
    * use `--resolution 2`
    * undistorted images and camera parameters are located under `<scene>/dense`

* **TUM**
  Only used in the scene separation study.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gottwald2025primu,
    title={PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage},
    author={Gottwald, Thomas and Heinert, Edgar and Stehr, Peter and Galappaththige, Chamuditha Jayanga and Rottmann, Matthias},
    journal={arXiv:2508.02443},
    year={2025}
}
```
