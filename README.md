Glacier Segmentation using U-Net++ (ResNet34, 5-Channel Data)

Deep learning pipeline for glacier region segmentation from multi-channel geospatial raster data using U-Net++ with ResNet34 encoder, strong augmentations, and MCC-based evaluation.

Designed for remote sensing, cryosphere studies, and climate research.

Project Overview

This project performs binary semantic segmentation to detect glacier regions from 5-channel satellite/derived raster inputs.

Key Highlights

U-Net++ architecture

ResNet34 encoder (ImageNet pretrained)

Combined BCE + Dice Loss

Matthews Correlation Coefficient (MCC) for evaluation

Heavy Albumentations-based augmentation

Automatic saving of best model

📂 Dataset Structure
DATA_DIR/
│
├── masked_sample1.npy      # 5-channel image (H, W, 5)
├── masked_sample2.npy
│
├── sample1.tif             # Corresponding binary mask
├── sample2.tif

🔹 Input Images
Property	Value
Format	.npy
Shape	H × W × 5
Type	float32
Description	Multi-spectral / derived raster bands
🔹 Masks
Property	Value
Format	.tif
Values	0 (non-glacier), 1 (glacier)
-> Model Architecture
Component	Details
Model	U-Net++
Encoder	ResNet34
Pretrained	ImageNet
Input Channels	5
Output Channels	1
Activation	None (logits)

Library Used

segmentation_models_pytorch

 Why ResNet34?

ResNet34 is a strong and efficient encoder for segmentation tasks.

Advantage	Benefit
 Faster training	Quicker experimentation
 Lower memory usage	Works on mid-range GPUs
 Stable gradients	Smooth convergence
 Strong baseline	Excellent for research tasks
 Data Augmentations

Training images are augmented using Albumentations:

Horizontal Flip

Vertical Flip

Random 90° Rotation

Affine Transform (scale, shear, rotate)

Random Brightness & Contrast

Improves robustness to terrain variation and orientation.

 Loss Function

Combined Loss = BCEWithLogits + Dice Loss


Loss=α⋅BCE+(1−α)⋅Dice
Parameter	Value
Alpha	0.7

✔ BCE → pixel accuracy
✔ Dice → region overlap

📊 Evaluation Metric
🧮 Matthews Correlation Coefficient (MCC)

MCC is ideal for imbalanced segmentation.

MCC∈[−1,1]
Score	Meaning
1	Perfect
0	Random
-1	Completely incorrect
⚙️ Training Configuration
Parameter	Value
Batch Size	4
Epochs	150
Optimizer	AdamW
Learning Rate	1e-4
Scheduler	ReduceLROnPlateau
Train/Val Split	80/20
Device	GPU (CUDA if available)
🔁 Training Workflow

Load .npy 5-channel images

Load .tif masks

Apply augmentations

Train U-Net++

Compute MCC per epoch

Save best model based on Validation MCC

💾 Model Saving

Best model is automatically saved:

MODEL_SAVE_PATH = ".../augmodel2.pth"

▶️ How to Run
1️⃣ Install Dependencies
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations opencv-python tqdm scikit-learn

2️⃣ Set Paths in Script
DATA_DIR = "path_to_dataset"
MODEL_SAVE_PATH = "path_to_save_model"

3️⃣ Start Training
python train.py

📈 Sample Output
Epoch 18 | Loss: 0.2145 | Train MCC: 0.84 | Val MCC: 0.80
🎉 Model saved (Best Val MCC: 0.80)

🔬 Applications

Glacier boundary extraction

Snow/ice segmentation

Climate change monitoring

Cryosphere mass balance analysis

