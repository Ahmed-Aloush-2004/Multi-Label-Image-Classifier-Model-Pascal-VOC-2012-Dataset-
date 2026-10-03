# Multi-Label Image Classification with PyTorch

An end-to-end **multi-label image classification** project using **PyTorch**, **ResNet18**, and the **Pascal VOC 2012** dataset.

Unlike traditional multi-class classification, where an image belongs to exactly one class, this project allows an image to contain **multiple classes simultaneously**.

For example, an image could contain:

```text
person → 90%
dog    → 85%
car    → 5%
```

The model can therefore predict multiple objects in the same image.

---

## 📌 Project Overview

This project demonstrates how to build a multi-label image classifier using:

- **PyTorch**
- **Torchvision**
- **ResNet18**
- **Pascal VOC 2012**
- **Binary Cross-Entropy Loss**
- **Sigmoid activation**
- **GPU acceleration with CUDA**

The complete workflow includes:

1. Importing libraries and checking GPU availability
2. Creating a custom Pascal VOC multi-label dataset
3. Converting XML annotations into multi-hot vectors
4. Applying image transformations
5. Creating PyTorch `DataLoader`s
6. Loading and modifying a pre-trained ResNet18
7. Training the model
8. Validating the model
9. Uploading custom images for inference
10. Detecting multiple classes using a probability threshold

---

# 1. Imports & GPU Check

First, we import the required libraries and determine whether a CUDA-enabled GPU is available.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


# Enable GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
```

### Why `xml.etree.ElementTree`?

The Pascal VOC dataset stores object annotations inside **XML files**.

These XML files contain information about the objects present in each image.

For example:

```xml
<object>
    <name>dog</name>
</object>

<object>
    <name>person</name>
</object>
```

We need to read these XML annotations and convert the detected object classes into a multi-hot binary vector.

---

# 2. Custom PyTorch Dataset Class

A standard classification dataset usually returns a single class ID:

```text
2
```

meaning:

```text
Class 2 = bird
```

However, multi-label classification needs to represent multiple classes at the same time.

For example:

```text
[1, 0, 1, 0]
```

could mean that classes `0` and `2` are present in the image.

Therefore, we create a custom PyTorch `Dataset` class that converts Pascal VOC annotations into multi-hot vectors.

---

## Pascal VOC Classes

Pascal VOC contains 20 official object classes:

```python
VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
```

---

## Custom Dataset

```python
class PascalVOCMultiLabel(Dataset):

    def __init__(self, root, image_set='train', transform=None):

        # Initialize the built-in Pascal VOC dataset downloader
        self.voc = datasets.VOCDetection(
            root=root,
            year='2012',
            image_set=image_set,
            download=True
        )

        self.transform = transform

        # Map class names to numerical indices
        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(VOC_CLASSES)
        }

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):

        img, target = self.voc[idx]

        # Create a zero-filled vector for all 20 classes
        label_vector = torch.zeros(
            len(VOC_CLASSES),
            dtype=torch.float32
        )

        # Extract object annotations
        objects = target['annotation']['object']

        # If there is only one object, convert it into a list
        if not isinstance(objects, list):
            objects = [objects]

        # Process every object in the image
        for obj in objects:

            cls_name = obj['name']

            if cls_name in self.class_to_idx:

                # Mark the class as present
                label_vector[
                    self.class_to_idx[cls_name]
                ] = 1.0

        # Apply image transformations
        if self.transform:
            img = self.transform(img)

        return img, label_vector
```

---

## 🔢 Why Use a Multi-Hot Vector?

In traditional single-label classification:

```text
2
```

could mean:

```text
bird
```

Only one class can be selected.

In multi-label classification:

```text
[0, 0, 1, 1, 0, ...]
```

could mean:

```text
bird = present
boat = present
```

Multiple classes can therefore be active simultaneously.

### Example

Suppose we have these four classes:

```text
0 = cat
1 = dog
2 = car
3 = person
```

An image containing a dog and a person would produce:

```text
[0, 1, 0, 1]
```

---

# 3. Transformations & DataLoaders

ResNet18 expects images with a consistent input size.

We resize the images to:

```text
224 × 224
```

We also normalize them using the standard ImageNet mean and standard deviation because the ResNet18 model was pre-trained on ImageNet.

```python
data_transforms = {

    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}
```

---

## Create Training and Validation Datasets

```python
train_dataset = PascalVOCMultiLabel(
    root='./data',
    image_set='train',
    transform=data_transforms['train']
)

val_dataset = PascalVOCMultiLabel(
    root='./data',
    image_set='val',
    transform=data_transforms['val']
)
```

---

## Create DataLoaders

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)
```

### DataLoader Configuration

| Parameter | Training | Validation |
|---|---:|---:|
| Batch size | 32 | 32 |
| Shuffle | Yes | No |
| Workers | 2 | 2 |

---

# 4. Model Architecture, Loss Function & Optimizer

We use a **pre-trained ResNet18** model.

Instead of training the entire network from scratch, we initially freeze the feature extractor and train only the final classification layer.

```python
# Load pre-trained ResNet18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze all feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features
num_features = model.fc.in_features

# Replace the original classification layer
# with a 20-class output layer
model.fc = nn.Linear(
    num_features,
    len(VOC_CLASSES)
)

# Move model to GPU/CPU
model = model.to(device)
```

---

## Loss Function

For multi-label classification, we use:

```python
criterion = nn.BCEWithLogitsLoss()
```

---

## Optimizer

Only the new fully connected layer is trained:

```python
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
```

---

# 5. Why `BCEWithLogitsLoss` Instead of `CrossEntropyLoss`?

This is one of the most important differences between **multi-class** and **multi-label** classification.

## Multi-Class Classification

For multi-class classification, an image belongs to exactly one class.

For example:

```text
cat
```

or:

```text
dog
```

but not both.

`CrossEntropyLoss` is commonly used for this task.

It works together with the concept of **Softmax**, where the class probabilities compete with each other and sum to approximately `1`.

For example:

```text
cat  = 0.90
dog  = 0.05
car  = 0.05
```

---

## Multi-Label Classification

In multi-label classification, several classes can be present simultaneously.

For example:

```text
person = 0.90
dog    = 0.85
car    = 0.05
```

These probabilities do not need to sum to `1`.

Each class is evaluated independently.

Therefore, we use:

```python
nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss` combines:

```text
Sigmoid + Binary Cross-Entropy
```

into one numerically stable operation.

---

# 6. Training and Validation Loop

We train the model for five epochs.

A probability threshold of `0.5` will later be used to determine whether a class is considered present.

```python
epochs = 5

threshold = 0.5
```

---

## Training Loop

```python
for epoch in range(epochs):

    # -------------------------
    # TRAINING PHASE
    # -------------------------

    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate BCE loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)


    # -------------------------
    # VALIDATION PHASE
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

    val_loss = val_loss / len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )
```

---

# 7. Prediction on Uploaded Images

After training, we can use the model to classify images from our computer.

The following code is designed for **Google Colab**.

```python
from google.colab import files
import io

print("Upload an image containing multiple objects:")

uploaded = files.upload()
```

---

## Inference Transform

The uploaded image must use the same basic preprocessing used during training.

```python
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
```

---

## Run Prediction

```python
for filename in uploaded.keys():

    # Load image
    raw_image = Image.open(
        io.BytesIO(uploaded[filename])
    ).convert("RGB")

    # Transform image and add batch dimension
    input_tensor = inference_transform(
        raw_image
    ).unsqueeze(0).to(device)

    # Evaluation mode
    model.eval()

    with torch.no_grad():

        # Get raw logits
        logits = model(input_tensor)

        # Convert logits into independent probabilities
        probabilities = torch.sigmoid(logits)[0]

    # Store detected classes
    detected_classes = []

    for idx, prob in enumerate(probabilities):

        if prob.item() >= 0.5:

            detected_classes.append(
                f"{VOC_CLASSES[idx]} "
                f"({prob.item() * 100:.1f}%)"
            )

    # Display result
    plt.figure(figsize=(6, 6))

    plt.imshow(raw_image)

    if detected_classes:

        title_str = (
            "Detected: "
            + ", ".join(detected_classes)
        )

    else:

        title_str = (
            "No classes detected (>50%)"
        )

    plt.title(
        title_str,
        fontsize=12
    )

    plt.axis("off")

    plt.show()
```

---

# 8. How Prediction Works

The model outputs **20 raw logits**:

```text
[logit_1, logit_2, ..., logit_20]
```

We apply the sigmoid function:

```python
probabilities = torch.sigmoid(logits)
```

This converts every logit independently into a probability between:

```text
0.0 → 1.0
```

For example:

```text
aeroplane → 0.03
bicycle   → 0.02
dog       → 0.91
person    → 0.87
car       → 0.12
```

Using a threshold of `0.5`:

```text
dog       → detected
person    → detected
aeroplane → not detected
bicycle   → not detected
car       → not detected
```

---

# 9. Multi-Class vs Multi-Label Classification

The most important differences can be summarized as follows:

| Feature | Multi-Class | Multi-Label |
|---|---|---|
| Number of classes per image | Usually one | Zero, one, or multiple |
| Target format | Single integer | Multi-hot vector |
| Example target | `2` | `[0, 1, 0, 1]` |
| Typical loss | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| Activation | Softmax | Sigmoid |
| Classes compete? | Yes | No |
| Prediction method | `argmax()` | Threshold |
| Probability sum | Approximately 1 | Not required to equal 1 |

---

# 10. Core Changes from Multi-Class to Multi-Label

The essential changes are:

### 1. Labels

Instead of:

```text
2
```

we use:

```text
[1, 0, 1, 0]
```

Each position independently represents whether a class is present.

---

### 2. Loss Function

Instead of:

```python
nn.CrossEntropyLoss()
```

we use:

```python
nn.BCEWithLogitsLoss()
```

---

### 3. Activation

Instead of:

```python
torch.softmax()
```

we use:

```python
torch.sigmoid()
```

Sigmoid is applied independently to each output neuron.

---

### 4. Prediction

Instead of:

```python
torch.argmax()
```

we use a threshold:

```python
probability >= 0.5
```

This allows multiple classes to be detected at the same time.

---

# 🧠 Complete Pipeline

The overall architecture can be visualized as:

```text
                 Pascal VOC 2012
                       │
                       ▼
              XML Object Annotations
                       │
                       ▼
              Extract Object Classes
                       │
                       ▼
             Multi-Hot Label Vector
             [1, 0, 1, 0, ..., 1]
                       │
                       │
Image ──► Resize ──► Normalize
                       │
                       ▼
                 ResNet18
              (Pre-trained)
                       │
                       ▼
                Fully Connected
                   20 Outputs
                       │
                       ▼
               Raw Logits
                       │
                       ▼
                 Sigmoid
                       │
                       ▼
             Independent Probabilities
                       │
                       ▼
              Threshold ≥ 0.5
                       │
                       ▼
              Multiple Predictions
```

---

# 📦 Requirements

Install the required packages with:

```bash
pip install torch torchvision pillow numpy matplotlib
```

If using Google Colab, most of these packages are already available.

---

# 🚀 Running the Project

## 1. Start the Python environment

You can run this project using:

- Google Colab
- Jupyter Notebook
- VS Code
- PyCharm
- Any Python environment with PyTorch installed

## 2. Run the imports

Execute the first section to verify your environment and GPU availability.

Expected output:

```text
Using device: cuda
```

if CUDA is available.

Otherwise:

```text
Using device: cpu
```

## 3. Create the datasets

The Pascal VOC 2012 dataset will be downloaded automatically when:

```python
download=True
```

is used.

The dataset will be stored under:

```text
./data
```

## 4. Train the model

Run the training section.

The model will print the training and validation loss after every epoch.

Example:

```text
Epoch 1/5 | Train Loss: 0.3214 | Val Loss: 0.2987
Epoch 2/5 | Train Loss: 0.2812 | Val Loss: 0.2671
...
```

## 5. Upload an image

After training, run the inference section and upload an image containing one or more recognizable objects.

The model will display the image along with the detected classes and their predicted probabilities.

---

# ⚠️ Important Notes

## This is Image Classification, Not Object Detection

Although Pascal VOC contains bounding-box annotations, this implementation does **not** draw bounding boxes around individual objects.

The model answers:

> "Which classes are present somewhere in this image?"

It does not answer:

> "Where exactly is each object located?"

For example, the output might be:

```text
person (94.2%)
dog (88.7%)
car (12.3%)
```

but it will not provide bounding boxes.

For actual object detection, architectures such as:

- Faster R-CNN
- YOLO
- RetinaNet
- SSD

would be more appropriate.

---

# 🔧 Possible Improvements

The current implementation is intentionally simple and can be extended with:

- Fine-tuning more ResNet18 layers
- Learning-rate scheduling
- Early stopping
- Data augmentation
- Class-weighted BCE loss
- Precision and recall metrics
- F1 score
- Mean Average Precision (mAP)
- Per-class performance analysis
- Confusion-style analysis for individual labels
- Model checkpoint saving
- Loading a previously trained model
- Better threshold selection for each class
- More advanced architectures such as ResNet50 or EfficientNet

---

# 📚 Key Concepts

This project demonstrates several important deep-learning concepts:

- Transfer learning
- CNN image classification
- Multi-label classification
- Multi-hot encoding
- Binary Cross-Entropy
- Sigmoid activation
- Pre-trained models
- Dataset customization
- XML annotation parsing
- PyTorch `Dataset`
- PyTorch `DataLoader`
- GPU acceleration
- Image preprocessing
- Model inference

---

# 📄 Summary

This project builds an end-to-end **multi-label image classifier** using:

```text
Pascal VOC 2012
       +
PyTorch
       +
Pre-trained ResNet18
       +
BCEWithLogitsLoss
       +
Sigmoid
       +
Probability Thresholding
```

The key idea is that **multiple classes can be active simultaneously**.

Instead of asking the model:

> "Which single class is this image?"

we ask:

> "Which of these 20 classes are present in this image?"

That distinction determines the choice of:

```text
Multi-hot labels
       ↓
BCEWithLogitsLoss
       ↓
Sigmoid
       ↓
Thresholding
```

rather than:

```text
Single class ID
       ↓
CrossEntropyLoss
       ↓
Softmax
       ↓
Argmax
```
