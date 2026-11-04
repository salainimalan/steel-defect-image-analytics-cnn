# Steel Surface Defect Classification with Image Analytics and CNNs

An educational image analytics project that studies steel surface images and compares classical image features, machine learning, and convolutional neural networks (CNNs) for six-class defect classification.

## Project at a glance

**Full project report:** [PROJECT_REPORT.md](PROJECT_REPORT.md) — project overview, methods, recorded results, and embedded plots.

- **Dataset:** NEU Surface Defect Database (NEU-DET)
- **Images:** 1,800 grayscale images, 200 × 200 pixels
- **Classes:** crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches
- **Data split:** 1,440 training images and 360 validation images
- **Main goal:** understand how image analytics can describe defect patterns and support CNN classification

The project includes a feature-fusion experiment that combines analytical image channels with a numerical feature table. Results depend on the training run, software versions, and hardware; reported scores should be treated as validation results for this dataset split, not as guarantees of performance in production.

## Defect categories

| Category | Typical visual pattern |
|---|---|
| Crazing | Fine, network-like cracks or lines |
| Inclusion | Irregular embedded or foreign-material regions |
| Patches | Broad areas with a different surface appearance |
| Pitted surface | Small pit-like marks or depressions |
| Rolled-in scale | Scale-like marks formed during rolling |
| Scratches | Thin, elongated surface marks |

## Analytics explored

- **Image representation:** RGB/grayscale inspection and comparisons of HSV and Lab color spaces. Although the images decode as three-channel files, they are visually close to grayscale, so grayscale analysis is central.
- **Intensity:** mean, standard deviation, minimum, maximum, and intensity distributions.
- **Edges:** Canny edge response and edge density.
- **Texture:** Local Binary Patterns (LBP), local variance, and Gray-Level Co-occurrence Matrix (GLCM) measures.
- **Frequency:** Discrete Cosine Transform (DCT) low- and high-frequency energy measurements.
- **Morphology:** thresholding, opening, closing, and connected-region descriptors. These provide exploratory structural cues; they are not reliable standalone defect masks for every image.

These measurements help compare categories and support classical ML. In the analytics-assisted CNN, grayscale, a high-frequency DCT representation, and a morphology response form three image channels. The Review 3 notebook adds the numerical analytical features through a separate MLP branch and joins both branches before classification.

## Repository layout

```text
.
├── dataset/                         # Local NEU-DET dataset (not tracked by Git)
├── notebooks/
│   ├── 00_neu_basic_image_analytics.ipynb
│   ├── 01_neu_dataset_exploration1.ipynb
│   ├── 02_neu_feature_analytics2.ipynb
│   ├── 04_neu_cnn_baseline.ipynb
│   ├── 05_neu_analytics_assisted_cnn.ipynb
│   └── 06_neu_review3_feature_fusion.ipynb
├── outputs/                          # Generated tables, plots, and model weights
└── create_neu_notebook_review3.py    # Generator for the Review 3 notebook
```

The current notebook numbering reflects the project history; there is no Notebook 03 file in this repository. Notebook 06 is the Review 3 feature-fusion experiment.

## Dataset setup

Obtain the NEU-DET dataset from its public distribution and extract it under `dataset/NEU-DET/`. The expected structure is:

```text
dataset/NEU-DET/
├── train/
│   ├── images/<class_name>/*.jpg
│   └── annotations/*.xml
└── validation/
    ├── images/<class_name>/*.jpg
    └── annotations/*.xml
```

Each split's `images` folder should contain one subfolder for each of the six class names listed above. The exploration notebook checks the files and annotation correspondence. Keep the dataset local: it is excluded from Git because of its size and licensing/distribution terms.

## Environment

The notebooks use Python with the following packages:

- OpenCV (`opencv-python`)
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch
- Jupyter Notebook or JupyterLab

Install any missing packages in the Python environment selected as the notebook kernel. PyTorch installation can vary by CPU/GPU setup; use the installation command recommended for your system by the official PyTorch installation selector.

## Run the notebooks

Open the repository in Jupyter, select the intended Python environment, and run notebooks in this order:

1. `00_neu_basic_image_analytics.ipynb` — representations, color models, histograms, and basic image operations.
2. `01_neu_dataset_exploration1.ipynb` — dataset inventory, samples, annotations, intensity summaries, DCT, and morphology demonstrations.
3. `02_neu_feature_analytics2.ipynb` — extracts the per-image feature table and category comparisons.
4. `04_neu_cnn_baseline.ipynb` — trains the grayscale CNN baseline.
5. `05_neu_analytics_assisted_cnn.ipynb` — trains a CNN using grayscale, DCT, and morphology image channels.
6. `06_neu_review3_feature_fusion.ipynb` — combines those image channels with the numerical feature table.

Notebooks 05 and 06 rely on outputs created by earlier notebooks. Run Notebook 02 before them so `outputs/neu_feature_analytics/neu_feature_table.csv` exists. Paths are resolved from the project directory or the `notebooks` directory; launch Jupyter from either location for the notebooks to locate the project.

## Feature table

Notebook 02 creates one row per image. The table includes identifiers (`split`, `category`, `file`) and 18 analytical columns:

| Feature group | Columns | What they describe |
|---|---|---|
| Intensity | `gray_mean`, `gray_std`, `gray_min`, `gray_max` | Brightness and its variation |
| Edges | `edge_density`, `edge_mean` | Amount and strength of edge response |
| Texture | `lbp_mean`, `lbp_std`, `local_variance_mean`, `local_variance_std` | Local patterns and neighborhood variation |
| GLCM | `glcm_contrast`, `glcm_homogeneity`, `glcm_energy` | Neighboring-pixel contrast and texture uniformity |
| DCT | `dct_low_energy_ratio`, `dct_high_energy_ratio` | Broad image structure and fine frequency detail |
| Morphology | `morphology_region_count`, `morphology_largest_area`, `morphology_foreground_ratio` | Thresholded connected-region structure |

Notebook 03's feature table is used by Logistic Regression and Random Forest (see the available model workflow in the repository's notebooks). Notebook 06 normalizes these numerical columns using training rows and feeds them to the feature branch of the fusion network. Labels and filenames are identifiers, not model input features.

## Model comparison

The repository explores three complementary approaches:

1. **Classical ML:** Logistic Regression and Random Forest classify the numerical feature table.
2. **Image-channel CNN:** the CNN learns from image pixels; the analytics-assisted version receives grayscale, high-frequency DCT, and morphology channels.
3. **Feature-fusion CNN:** a CNN image branch and a dense numerical-feature branch are concatenated and passed to a six-class classifier.

Use the generated confusion matrices and accuracy summaries to compare class-specific errors. A high validation score on NEU-DET does not establish robustness to other cameras, lighting, steel products, or factory conditions.

## Generated outputs

Notebook outputs are written under `outputs/`, including:

- Dataset inventory, annotation checks, and sample visualizations
- Feature table and feature-comparison plots
- Classical ML feature importance and confusion matrices
- CNN training histories and confusion matrices
- Model comparison plots and feature-fusion results

`dataset/` and `outputs/` are ignored by Git by default. The repository currently tracks selected PNG plots as illustrative results; regenerate tables, metrics, and model weights by running the notebooks.

## Reproducibility notes

- CNN training is stochastic; results may vary across runs and hardware.
- The validation set is used for model comparison, so these scores are validation measurements.
- The numerical feature scaler is fitted on the training split in the feature-fusion notebook.
- Ground-truth annotations are for dataset exploration and evaluation; they must not be used as input to feature construction or defect prediction.
- Morphology responses depend on thresholding and kernel choices and should be interpreted as exploratory features.

## License and dataset attribution

This repository contains project notebooks and selected result plots. It does not include the NEU-DET dataset. Follow the dataset provider's terms for downloading, use, and redistribution, and add the appropriate attribution and license information before redistributing dataset-derived materials.
