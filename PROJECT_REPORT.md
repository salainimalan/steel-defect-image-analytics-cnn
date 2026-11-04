# Steel Surface Defect Classification with Image Analytics and CNNs

## Project Report

## Abstract

This project applies image analytics and convolutional neural networks to classify steel surface images from the NEU Surface Defect Database (NEU-DET). The notebooks inspect image representation and dataset annotations, calculate intensity, edge, texture, DCT, and morphology features, and evaluate CNNs with different image inputs. The final notebook combines three analytical image channels with the numerical feature table. The recorded validation results are 88.61% best epoch accuracy for the grayscale CNN, 98.33% for the analytics-assisted CNN, and 98.61% best epoch accuracy for the feature-fusion CNN. These values are taken from the current CNN notebook outputs.

## 1. Introduction

Steel surface defects can appear as cracks, inclusions, patches, pits, scale patterns, or scratches. This project examines how digital image processing can describe these visual differences and how a CNN can use the resulting image information for classification.

The project is image-level classification. Each image is assigned one of six defect categories. The bounding-box annotations are inspected in the dataset exploration notebook, but the CNN notebooks classify the image and do not perform defect localization.

## 2. Project objectives

- Explore image representation using grayscale, RGB, HSV, and Lab views.
- Inspect dataset images, intensity distributions, and annotation correspondence.
- Demonstrate DCT and morphological image processing.
- Extract numerical intensity, edge, texture, frequency, and morphology measurements.
- Train and evaluate a grayscale CNN.
- Train and evaluate a CNN with grayscale, DCT, and morphology channels.
- Combine image channels and numerical features in a feature-fusion CNN.

## 3. Dataset

The inspected NEU-DET data contains 1,800 images of size 200 × 200 pixels. There are 240 training images and 60 validation images in each of six classes. The images are JPG files and visually close to grayscale.

| Defect category | Typical appearance |
|---|---|
| Crazing | Fine network-like cracks or lines |
| Inclusion | Irregular embedded regions |
| Patches | Broad areas with a different appearance |
| Pitted surface | Small pit-like marks |
| Rolled-in scale | Scale-like surface marks |
| Scratches | Thin elongated marks |

Notebook 01 checks Pascal VOC XML annotation correspondence. All validation images have a matching annotation. One training image, `crazing_240.jpg`, has no corresponding XML file. Sample bounding boxes are displayed for visual inspection.

## 4. Notebook methods and findings

### 4.1 Notebook 00 — Basic Image Analytics

This notebook displays RGB, grayscale, HSV, and Lab representations of a sample image. It plots separate RGB channel histograms and a grayscale histogram, then demonstrates Gaussian blur, Canny edge detection, and grayscale normalization. The images appear close to grayscale, so grayscale is the primary representation used in later analysis. The histograms describe pixel brightness distribution, and Canny shows local edges and transitions.

### 4.2 Notebook 01 — Dataset Exploration

The inventory cell records image counts, dimensions, channels, and formats for each split and class. Annotation correspondence is checked by matching image filenames with XML files. Representative images are displayed with their bounding boxes. Per-image grayscale mean, standard deviation, minimum, and maximum are calculated and summarized by class.

The notebook also demonstrates morphology and DCT. Otsu thresholding creates a binary image, then opening and closing alter small regions and gaps. The resulting response remains irregular due to the steel texture, so the demonstration is not a reliable defect segmentation method. DCT visualization separates low-frequency structure from high-frequency detail, illustrating the different spatial information represented in the transform.

### 4.3 Notebook 02 — Image Feature Analytics

The feature extraction notebook creates one row per image and 18 numerical measurements:

| Feature group | Feature columns | Description from the image measurements |
|---|---|---|
| Intensity | `gray_mean`, `gray_std`, `gray_min`, `gray_max` | Grayscale brightness and variation |
| Edges | `edge_density`, `edge_mean` | Canny edge fraction and mean response |
| Texture | `lbp_mean`, `lbp_std` | Summary statistics of local binary patterns |
| Local variation | `local_variance_mean`, `local_variance_std` | Neighborhood intensity variation |
| GLCM | `glcm_contrast`, `glcm_homogeneity`, `glcm_energy` | Gray-level relationships between horizontal neighbors |
| DCT | `dct_low_energy_ratio`, `dct_high_energy_ratio` | Proportions of DCT energy in low and high frequencies |
| Morphology | `morphology_region_count`, `morphology_largest_area`, `morphology_foreground_ratio` | Connected-region count, largest area, and foreground fraction |

The notebook displays category summaries and boxplots for grayscale mean, edge density, local variance, LBP mean, and DCT high-frequency energy. The comparisons show that the classes differ across multiple image properties. The morphology values are derived from thresholded images and should be interpreted as structural measurements of those responses.

### 4.4 Notebook 04 — Grayscale CNN Baseline

The baseline loads images as grayscale, converts them to tensors, and normalizes them. Its CNN uses two convolution and max-pooling stages, followed by fully connected classification layers. It is trained for ten epochs using cross-entropy loss and the Adam optimizer. The saved training history has a best validation accuracy of 88.61% at epoch 7; the final epoch records 80.83%. The confusion matrix shows the validation predictions by class.

### 4.5 Notebook 05 — Analytics-Assisted CNN

This CNN receives three channels derived from each image:

1. Normalized grayscale intensity.
2. Normalized high-frequency DCT magnitude after low-frequency coefficients are removed.
3. An Otsu-thresholded morphology response refined with opening and closing.

The model uses a two-stage CNN and a six-class output layer. Training uses cross-entropy loss and Adam for ten epochs. Its saved history records 98.33% validation accuracy at epoch 10, and the confusion matrix shows its class predictions.

### 4.6 Notebook 06 — CNN with Analytical Feature Fusion

The feature-fusion model uses the same three image channels as Notebook 05 and adds the 18 numerical features created by Notebook 02. The image branch is a CNN. The numerical branch is a dense layer followed by ReLU. The two branch outputs are concatenated and sent to the final six-class classifier. The feature scaler is fitted using training rows before transforming the feature values.

The saved history records its best validation accuracy, 98.61%, at epoch 9; the final epoch records 98.06%. The confusion matrix displays predictions from the final validation evaluation.

## 5. Model results

| CNN model | Model input | Best recorded validation accuracy | Final epoch accuracy |
|---|---|---:|---:|
| Grayscale CNN | Grayscale image | 88.61% at epoch 7 | 80.83% |
| Analytics-Assisted CNN | Grayscale, DCT, and morphology channels | 98.33% at epoch 10 | 98.33% |
| Feature-Fusion CNN | Three image channels and 18 numerical features | 98.61% at epoch 9 | 98.06% |

The feature-fusion CNN has the highest best-epoch score among the three CNNs in the recorded histories. Its best score is 0.28 percentage points higher than the analytics-assisted CNN. Its final epoch is 0.27 percentage points below the analytics-assisted CNN. The grayscale CNN has lower validation accuracy in its recorded run. These comparisons describe the saved outputs for the current runs.

## 6. Figures

### Feature analytics

![Feature distributions across defect categories](outputs/neu_feature_analytics/feature_comparisons.png)

### Grayscale CNN

![Grayscale CNN validation confusion matrix](outputs/neu_cnn/cnn_confusion_matrix.png)

### Analytics-assisted CNN

![Analytics-assisted CNN validation confusion matrix](outputs/neu_analytics_cnn/analytics_cnn_confusion_matrix.png)

### Feature-fusion CNN

![Feature-fusion CNN validation confusion matrix](outputs/neu_review3_feature_fusion/fusion_confusion_matrix.png)

## 7. Conclusion

The notebooks establish an image analytics workflow for the NEU-DET steel surface dataset. They inspect grayscale representations, calculate numerical intensity, edge, texture, DCT, and morphology properties, and use analytical image channels in CNN classification. In the current recorded runs, the analytics-assisted CNN and feature-fusion CNN achieve higher validation accuracy than the grayscale CNN. The fusion CNN records the highest best-epoch accuracy, while its final-epoch result is slightly below that of the analytics-assisted CNN. The saved results therefore show that both analytical channels and numerical features are used in the project’s CNN experiments.

## 8. Reproducibility

Place NEU-DET at `dataset/NEU-DET/`, with `train` and `validation` folders containing class-organized `images` and corresponding `annotations`. Install the packages imported by the notebooks in the chosen Jupyter kernel. Run notebooks 00, 01, and 02 before notebooks 04, 05, and 06. Notebook 02 creates the feature table required by Notebook 06. Generated files are written under `outputs/`.
