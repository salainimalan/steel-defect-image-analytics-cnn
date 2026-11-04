# Steel Surface Defect Classification with Image Analytics and CNNs

This repository contains the current notebooks and selected output plots for a NEU-DET steel surface image classification project. The work explores image representation, image analytics, and CNN-based classification.

## Dataset

NEU-DET has 1,800 images at 200 × 200 pixels across six categories. The project uses 1,440 training images and 360 validation images, with 240 training and 60 validation images per class. The images are visually close to grayscale.

Classes: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.

Place the extracted dataset under `dataset/NEU-DET/`:

```text
dataset/NEU-DET/
├── train/images/<class_name>/*.jpg
├── train/annotations/*.xml
├── validation/images/<class_name>/*.jpg
└── validation/annotations/*.xml
```

The dataset is not included in this repository.

## Notebooks

Run in this order:

1. `00_neu_basic_image_analytics.ipynb` — compares RGB, grayscale, HSV, and Lab; plots histograms; demonstrates blur, edges, and normalization.
2. `01_neu_dataset_exploration1.ipynb` — inventories images, checks annotation correspondence, summarizes intensity, and demonstrates morphology and DCT.
3. `02_neu_feature_analytics2.ipynb` — extracts intensity, edge, texture, GLCM, DCT, and morphology features; exports a feature table and comparison plot.
4. `04_neu_cnn_baseline.ipynb` — trains and evaluates a grayscale CNN.
5. `05_neu_analytics_assisted_cnn.ipynb` — trains a CNN using grayscale, high-frequency DCT, and morphology image channels.
6. `06_neu_review3_feature_fusion.ipynb` — combines those image channels with the numerical feature table.

There is no Notebook 03 in the current repository.

## Feature table

Notebook 02 generates one row per image. Alongside `split`, `category`, and `file`, it contains 18 measurements:

| Group | Columns |
|---|---|
| Intensity | `gray_mean`, `gray_std`, `gray_min`, `gray_max` |
| Edges | `edge_density`, `edge_mean` |
| Texture | `lbp_mean`, `lbp_std`, `local_variance_mean`, `local_variance_std` |
| GLCM | `glcm_contrast`, `glcm_homogeneity`, `glcm_energy` |
| DCT | `dct_low_energy_ratio`, `dct_high_energy_ratio` |
| Morphology | `morphology_region_count`, `morphology_largest_area`, `morphology_foreground_ratio` |

Notebook 06 uses the 18 measurements as its numerical input branch. Its feature scaler is fitted on training rows.

## Current CNN inputs

- **Grayscale CNN:** one normalized grayscale image channel.
- **Analytics-assisted CNN:** grayscale, high-frequency DCT response, and morphology response as three image channels.
- **Feature-fusion CNN:** the same three image channels plus the numerical feature vector.

The morphology response is an exploratory structural representation; thresholding does not produce a reliable defect mask for every image.

## Run

Use a Python environment with OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, PyTorch, and Jupyter installed. Run Notebook 02 before Notebook 06 because it creates `outputs/neu_feature_analytics/neu_feature_table.csv`.

Generated tables, plots, and model files are written to `outputs/`. The dataset and generated outputs are ignored by Git by default; selected PNG plots are tracked in this repository.

## Contributors

| **Name** | **Roll No** |
|---|---|
| **Salai Nimalan** | 23MIA1064 |
| **Aditya Rao B** | 23MIA1120 |
