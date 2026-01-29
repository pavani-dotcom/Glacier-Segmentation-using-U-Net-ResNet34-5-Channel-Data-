import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from segmentation_models_pytorch import UnetPlusPlus
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


DATA_DIR = "/content/drive/MyDrive/Glacier Hack Atmpts/Atmpt 3/augmented dataset"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Glacier Hack Atmpts/Attempt04/augmodel2.pth"
BATCH_SIZE = 4
EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])



class GlacierDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        
        all_files = os.listdir(root_dir)
        self.npy_files = [f for f in all_files if f.endswith('.npy') and f.startswith('masked_')]

        
        if indices is not None:
            self.npy_files = [self.npy_files[i] for i in indices]

        self.samples = []
        for f in self.npy_files:
            base_name = f.replace('masked_', '').replace('.npy', '')
            mask_path = os.path.join(root_dir, base_name + '.tif')
            if os.path.exists(mask_path):
                self.samples.append((os.path.join(root_dir, f), mask_path))
            else:
                print(f" Mask not found: {mask_path}")

        print(f" Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        
        image = np.load(img_path).astype(np.float32)  # H, W, 5
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot load: {mask_path}")
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)

        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # add channel for mask
        else:
            image = torch.tensor(image.transpose(2, 0, 1))
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
        
        

model = UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights="imagenet",
    in_channels=5,
    classes=1,
    activation=None
).to(DEVICE)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

criterion = CombinedLoss(alpha=0.7)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)



def compute_mcc(pred, target):
    pred = (pred > 0.5).astype(int).flatten()
    target = target.astype(int).flatten()
    return matthews_corrcoef(target, pred)


# DATA SPLIT

# full dataset
full_dataset = GlacierDataset(DATA_DIR)

# split indices
train_idx, val_idx = train_test_split(list(range(len(full_dataset.samples))), test_size=0.2, random_state=42)

# train/val datasets
train_dataset = GlacierDataset(DATA_DIR, indices=train_idx, transform=train_transform)
val_dataset = GlacierDataset(DATA_DIR, indices=val_idx, transform=val_transform)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# TRAINING LOOP

best_mcc = -1.0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, masks in progress_bar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        targets = masks.cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets)

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])
    train_mcc = compute_mcc(all_preds, all_targets)
    avg_loss = total_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()
            val_preds.append(preds)
            val_targets.append(targets)

    val_preds = np.concatenate([p.flatten() for p in val_preds])
    val_targets = np.concatenate([t.flatten() for t in val_targets])
    val_mcc = compute_mcc(val_preds, val_targets)

    print(f"\n Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train MCC: {train_mcc:.4f} | Val MCC: {val_mcc:.4f}\n")

    # save best
    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" Model saved (Best Val MCC: {best_mcc:.4f})")

    scheduler.step(val_mcc)

print(f" Training finished! Best Val MCC: {best_mcc:.4f}")
print(f" Model saved to: {MODEL_SAVE_PATH}")
