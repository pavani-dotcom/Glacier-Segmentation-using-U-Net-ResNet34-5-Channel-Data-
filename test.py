import os
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import UnetPlusPlus
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm



MODEL_PATH = "/content/drive/MyDrive/Glacier Hack Atmpts/Attempt04/augmodel2.pth"
TEST_IMAGE_DIR = "/content/drive/MyDrive/Glacier Hack Atmpts/Atmpt 1/actual dataset"
TEST_LABEL_DIR = "/content/drive/MyDrive/Glacier Hack Atmpts/Atmpt 1/actual dataset/label"
BANDS = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']
BAND_PREFIX_MAP = {
    'Band1': 'B2_B2_masked',
    'Band2': 'B3_B3_masked',
    'Band3': 'B4_B4_masked',
    'Band4': 'B6_B6_masked',
    'Band5': 'B10_B10_masked'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL
model = UnetPlusPlus(
    encoder_name="efficientnet-b5",
    encoder_weights="imagenet",
    in_channels=5,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(" Model loaded.")



def tta_inference(model, image):
    preds = []
    with torch.no_grad():
        # Original
        p = torch.sigmoid(model(image))
        preds.append(p)
        # Horizontal flip
        p = torch.sigmoid(model(torch.flip(image, dims=[3])))
        preds.append(torch.flip(p, dims=[3]))
        # Vertical flip
        p = torch.sigmoid(model(torch.flip(image, dims=[2])))
        preds.append(torch.flip(p, dims=[2]))
        # H+V flip
        p = torch.sigmoid(model(torch.flip(image, dims=[2,3])))
        preds.append(torch.flip(p, dims=[2,3]))
    pred = torch.mean(torch.stack(preds, dim=0), dim=0)
    return pred.squeeze().cpu().numpy()



band1_dir = os.path.join(TEST_IMAGE_DIR, 'Band1')
prefix = BAND_PREFIX_MAP['Band1']
image_ids = []

for f in os.listdir(band1_dir):
    if f.startswith(prefix) and f.endswith('.tif'):
        suffix = f.replace(prefix + '_', '').replace('.tif', '')
        image_ids.append(suffix)

print(f" Found {len(image_ids)} test images.")



images = []
masks = []

for suffix in tqdm(image_ids, desc="Loading data"):
    try:
        
        bands = []
        for band_folder, prefix in BAND_PREFIX_MAP.items():
            filename = f"{prefix}_{suffix}.tif"
            path = os.path.join(TEST_IMAGE_DIR, band_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if img.max() > 0:
                img = img / img.max()
            bands.append(img)
        image = np.stack(bands, axis=-1)  # H x W x 5
        image = torch.tensor(image.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
        images.append((suffix, image))

       
        mask_path = os.path.join(TEST_LABEL_DIR, f"Y_output_resized_{suffix}.tif")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        mask = (mask > 127).astype(int)
        masks.append(mask.flatten())

    except Exception as e:
        print(f" Skip {suffix}: {e}")
        continue


best_thresh = 0.5
best_mcc = -1.0

thresholds = np.arange(0.3, 0.71, 0.01)

for t in thresholds:
    all_preds = []
    for suffix, image in images:
        pred = tta_inference(model, image)
        all_preds.extend((pred > t).astype(int).flatten())
    all_targets = np.concatenate(masks)
    mcc = matthews_corrcoef(all_targets, all_preds)
    if mcc > best_mcc:
        best_mcc = mcc
        best_thresh = t

print(f"\n Best threshold: {best_thresh:.2f} | MCC: {best_mcc:.4f}")



os.makedirs("/content/drive/MyDrive/glacier hack/submission2", exist_ok=True)

for suffix, image in tqdm(images, desc="Saving predictions"):
    pred = tta_inference(model, image)
    pred = (pred > best_thresh).astype(np.uint8) * 255
    output_path = os.path.join("/content/drive/MyDrive/glacier hack/submission2", f"{suffix}.png")
    cv2.imwrite(output_path, pred)

print(" All predictions saved with optimal threshold!")
