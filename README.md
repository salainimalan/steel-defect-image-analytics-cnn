# Steel Surface Defect Classification Using Image Analytics and CNNs

An image analytics and deep-learning project for classifying steel surface defects in the NEU Surface Defect Database (NEU-DET). The workflow studies intensity, edges, texture, frequency-domain information, and morphology, then compares classical machine learning with CNN-based approaches.

## Project overview

- **Dataset:** NEU Surface Defect Database (NEU-DET)
- **Images:** 1,800 grayscale-looking images, each 200 × 200 pixels
- **Classes:** crazing, inclusion, patches, pitted surface, rolled-in scale, scratches
- **Split:** 1,440 training images and 360 validation images
- **Task:** Image-level classification into six defect categories

The dataset is not included in this repository. The project report is available in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Defect categories

| Class | Typical surface pattern |
|---|---|
| Crazing | Fine network-like cracks or lines |
| Inclusion | Irregular embedded or foreign-material regions |
| Patches | Broad areas with a different appearance |
| Pitted surface | Small pit-like marks or depressions |
| Rolled-in scale | Scale-like surface marks |
| Scratches | Thin elongated marks |

## Analysis workflow

The notebooks progress from image representation to feature analysis and classification:

1. `00_neu_basic_image_analytics.ipynb` — RGB, grayscale, HSV and Lab representations; histograms; blur, edges, normalization, and basic statistics.
2. `01_neu_dataset_exploration1.ipynb` — dataset inventory, annotation matching, sample images, intensity summaries, morphology and DCT demonstrations.
3. `02_neu_feature_analytics2.ipynb` — per-image intensity, edge, LBP, local variance, GLCM, DCT, and morphology features.
4. `04_neu_cnn_baseline.ipynb` — grayscale CNN baseline.
5. `05_neu_analytics_assisted_cnn.ipynb` — CNN using grayscale, high-frequency DCT, and morphology image channels.
6. `06_neu_review3_feature_fusion.ipynb` — CNN image branch fused with the numerical feature table.

The numbering follows the existing notebook files; there is no Notebook 03 in the repository.

## Models and recorded results

| Model | Input | Recorded validation accuracy |
|---|---|---:|
| Logistic Regression | Numerical feature table | 95.83% |
| Random Forest | Numerical feature table | 98.33% |
| Grayscale CNN | Grayscale image | 88.61% best recorded epoch |
| Analytics-Assisted CNN | Grayscale, DCT, and morphology channels | 98.33% |
| Feature-Fusion CNN | Three image channels plus numerical feature vector | 98.61% best epoch; 98.06% final epoch |

The values are from recorded runs on the same 360-image validation split. They are not independent test-set estimates. The small difference between feature fusion and the other strong models should be interpreted cautiously.

## Image analytics

Notebook 02 creates 18 numerical features:

| Group | Features | Description |
|---|---|---|
| Intensity | `gray_mean`, `gray_std`, `gray_min`, `gray_max` | Brightness and variation |
| Edges | `edge_density`, `edge_mean` | Amount and strength of edge response |
| Texture | `lbp_mean`, `lbp_std`, `local_variance_mean`, `local_variance_std` | Local patterns and neighborhood variation |
| GLCM | `glcm_contrast`, `glcm_homogeneity`, `glcm_energy` | Relationships between neighboring pixel intensities |
| DCT | `dct_low_energy_ratio`, `dct_high_energy_ratio` | Broad structure and fine-frequency detail |
| Morphology | `morphology_region_count`, `morphology_largest_area`, `morphology_foreground_ratio` | Thresholded connected-region properties |

Morphological responses provide exploratory structural information. They do not reliably isolate defects in every image and are not treated as ground-truth segmentation masks.

## Dataset setup

Extract NEU-DET under `dataset/NEU-DET/` with the following structure:

```text
dataset/NEU-DET/
├── train/
│   ├── images/<class_name>/*.jpg
│   └── annotations/*.xml
└── validation/
    ├── images/<class_name>/*.jpg
    └── annotations/*.xml
```

Each `images` folder should contain subfolders named `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`.

## Environment and execution

The notebooks use Python with OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, PyTorch, and Jupyter. Install these packages in the selected notebook kernel; choose a PyTorch build suitable for the local CPU/GPU.

Run notebooks in the order listed above. Notebook 02 generates `outputs/neu_feature_analytics/neu_feature_table.csv`, which Notebook 06 requires. The notebooks write generated tables, plots, training histories, and model weights under `outputs/`.

## Selected plots

- Basic representations and operations: `outputs/neu_basic_analytics/`
- Dataset samples, DCT, and morphology: `outputs/neu_dataset_exploration/`
- Feature comparisons: `outputs/neu_feature_analytics/`
- CNN confusion matrices: `outputs/neu_cnn/`, `outputs/neu_analytics_cnn/`, and `outputs/neu_review3_feature_fusion/`

The dataset, generated tables, and model weights are excluded from Git. Selected PNG plots are tracked. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed methodology, findings, figures, and interpretation.

## Contributors

| **Name** | **Roll No** |
|---|---|
| **Salai Nimalan** | 23MIA1064 |
| **Aditya Rao B** | 23MIA1120 |
