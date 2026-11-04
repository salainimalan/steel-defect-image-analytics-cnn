import json
from pathlib import Path

def md(title):
    return {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.strip().splitlines()]}

cells = [
    md("Review 3: CNN with Analytical Feature Fusion"),
    md("This notebook combines the three analytical image channels with the numerical feature table."),
    code('''from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PROJECT = Path.cwd()
if PROJECT.name == "notebooks":
    PROJECT = PROJECT.parent
ROOT = PROJECT / "dataset" / "NEU-DET"
FEATURE_FILE = PROJECT / "outputs" / "neu_feature_analytics" / "neu_feature_table.csv"
OUTPUT = PROJECT / "outputs" / "neu_review3_feature_fusion"
OUTPUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
FEATURES = ["gray_mean", "gray_std", "gray_min", "gray_max", "edge_density", "edge_mean", "lbp_mean", "lbp_std", "local_variance_mean", "local_variance_std", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "dct_low_energy_ratio", "dct_high_energy_ratio", "morphology_region_count", "morphology_largest_area", "morphology_foreground_ratio"]
device = "cuda" if torch.cuda.is_available() else "cpu"
'''),
    md("Understanding the Feature Table"),
    code('''table = pd.read_csv(FEATURE_FILE)
print("Rows:", len(table))
print("Analytical features:", len(FEATURES))
display(table.groupby("category")[FEATURES].mean().round(3))
'''),
    md("Train-Only Feature Normalization"),
    code('''scaler = StandardScaler()
scaler.fit(table.loc[table["split"] == "train", FEATURES])
table[FEATURES] = scaler.transform(table[FEATURES])
'''),
    md("Image Channels and Feature Vectors"),
    code('''def channels(file):
    gray = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (200, 200)).astype(np.float32) / 255
    dct = cv2.dct(gray)
    high = dct.copy()
    high[:32, :32] = 0
    high = cv2.normalize(np.abs(high), None, 0, 1, cv2.NORM_MINMAX)
    binary = cv2.threshold((gray * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    morphology = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255
    return np.stack([gray, high, morphology])

def feature_row(file):
    return table.loc[table["file"] == file.name, FEATURES].iloc[0].to_numpy(dtype=np.float32)
'''),
    md("Dataset and Fusion Model"),
    code('''class FusionDataset(Dataset):
    def __init__(self, split):
        rows = table[table["split"] == split]
        self.items = [(ROOT / split / "images" / row["category"] / row["file"], CLASSES.index(row["category"]), row[FEATURES].to_numpy(dtype=np.float32)) for _, row in rows.iterrows()]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        file, label, features = self.items[index]
        return torch.tensor(channels(file)), torch.tensor(features), label

class FusionCNN(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.image_branch = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.image_head = nn.Sequential(nn.Linear(32 * 50 * 50, 64), nn.ReLU())
        self.feature_head = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU())
        self.classifier = nn.Linear(96, 6)
    def forward(self, image, features):
        image_vector = self.image_head(self.image_branch(image))
        feature_vector = self.feature_head(features)
        return self.classifier(torch.cat([image_vector, feature_vector], dim=1))

train_loader = DataLoader(FusionDataset("train"), batch_size=32, shuffle=True)
validation_loader = DataLoader(FusionDataset("validation"), batch_size=32)
model = FusionCNN(len(FEATURES)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
'''),
    md("Training and Evaluation"),
    code('''history = []
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, features, labels in train_loader:
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images, features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, features, labels in validation_loader:
            actual.extend(labels.numpy())
            predicted.extend(model(images.to(device), features.to(device)).argmax(1).cpu().numpy())
    accuracy = np.mean(np.array(actual) == np.array(predicted))
    history.append({"epoch": epoch + 1, "loss": total_loss / len(train_loader), "validation_accuracy": accuracy})
    print(history[-1])

print(classification_report(actual, predicted, target_names=CLASSES))
ConfusionMatrixDisplay.from_predictions(actual, predicted, display_labels=CLASSES, xticks_rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT / "fusion_confusion_matrix.png", dpi=150)
plt.show()
pd.DataFrame(history).to_csv(OUTPUT / "fusion_history.csv", index=False)
torch.save(model.state_dict(), OUTPUT / "fusion_cnn.pt")
'''),
    md("Review Findings"),
    code('''print("The CNN branch learns visual patterns from grayscale, DCT, and morphology channels.")
print("The analytical branch supplies normalized intensity, edge, texture, DCT, and morphology measurements.")
print("Feature fusion allows the final classifier to use learned image patterns and interpretable measurements together.")
''')
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (base)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/06_neu_review3_feature_fusion.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
