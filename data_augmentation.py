import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


DATASET_DIR = "/content/drive/MyDrive/Glacier Hack/Atmpt 1/actual dataset"  
OUTPUT_DIR = "/content/drive/MyDrive/Glacier Hack/Atmpt 2/augmented dataset"  
NUM_AUGMENTS_PER_IMAGE = 8      

BAND_PREFIX_MAP = {
    'Band1': 'B2_B2_masked',
    'Band2': 'B3_B3_masked',
    'Band3': 'B4_B4_masked',
    'Band4': 'B6_B6_masked',
    'Band5': 'B10_B10_masked'
}

LABEL_DIR = "label"

 
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(noise_var_limit=(10.0, 50.0), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(alpha=100, sigma=50, p=0.3),

], additional_targets={'mask': 'mask'})


os.makedirs(OUTPUT_DIR, exist_ok=True)



band1_dir = os.path.join(DATASET_DIR, 'Band1')
prefix = BAND_PREFIX_MAP['Band1']
suffixes = []

for f in os.listdir(band1_dir):
    if f.startswith(prefix) and f.endswith('.tif'):
        suffix = f.replace(prefix + '_', '').replace('.tif', '')
        suffixes.append(suffix)

print(f" Found {len(suffixes)} images. Example: {suffixes[:5]}")



count = 0

for suffix in tqdm(suffixes, desc="Augmenting"):
    try:
        
        bands = []
        for band_folder, prefix in BAND_PREFIX_MAP.items():
            filename = f"{prefix}_{suffix}.tif"
            path = os.path.join(DATASET_DIR, band_folder, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing band: {path}")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load: {path}")
            img = img.astype(np.float32)
            if img.max() > 0:
                img = img / img.max()
            bands.append(img)

       
        image = np.stack(bands, axis=-1)

       
        mask_path = os.path.join(DATASET_DIR, LABEL_DIR, f"{suffix}.tif")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(DATASET_DIR, LABEL_DIR, f"Y_output_{suffix}.tif")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {suffix}")

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] 

       
        for i in range(NUM_AUGMENTS_PER_IMAGE):
            augmented = augmentation_pipeline(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            
            np.save(os.path.join(OUTPUT_DIR, f"masked_{suffix}_aug_{i}.npy"), aug_image.astype(np.float32))

            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{suffix}_aug_{i}.tif"), aug_mask.astype(np.uint8))

            count += 1

    except Exception as e:
        print(f" Skip {suffix}: {e}")
        continue

print(f" Generated {count} augmented samples in {OUTPUT_DIR}")