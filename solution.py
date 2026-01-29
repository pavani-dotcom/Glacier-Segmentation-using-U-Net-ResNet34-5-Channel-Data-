# pip install torch torchvision tifffile  

import os
import numpy as np
import torch
import argparse
import tifffile
import cv2
import json
import re


BANDS = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model_path = "/work/model.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)
    return model



def tta_inference(model, image):
    """ Apply TTA (flips) and return averaged prediction [H, W]. """
    preds = []

    with torch.no_grad():
        # Original
        p = torch.sigmoid(model(image))
        preds.append(p)

        # Horizontal flip
        #p = torch.sigmoid(model(torch.flip(image, dims=[3])))
        #preds.append(torch.flip(p, dims=[3]))

        # Vertical flip
        #p = torch.sigmoid(model(torch.flip(image, dims=[2])))
        #preds.append(torch.flip(p, dims=[2]))

        # H+V flip
        p = torch.sigmoid(model(torch.flip(image, dims=[2, 3])))
        preds.append(torch.flip(p, dims=[2, 3]))

    pred = torch.mean(torch.stack(preds, dim=0), dim=0)
    return pred.squeeze().cpu().numpy()


def maskgeration(imagepath, _ignored_output=None):
   
    out_dir = "/work/predictions"
    os.makedirs(out_dir, exist_ok=True)

    model = load_model()
    saved_masks = {}

    band1_dir = imagepath['Band1']
    image_ids = [f for f in os.listdir(band1_dir) if f.endswith(".tif")]

    for fname in image_ids:
        try:
            bands = []
            for b in BANDS:
                path = os.path.join(imagepath[b], fname)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Failed to load {path}")
                img = img.astype(np.float32)
                if img.max() > 0:
                    img = img / img.max()
                bands.append(img)

            image = np.stack(bands, axis=-1)
            image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

           
            pred = tta_inference(model, image)

            
            pred = (pred > 0.5).astype(np.uint8) * 255

            
            out_path = os.path.join(out_dir, fname)  
            tifffile.imwrite(out_path, pred)

            
            tile_id_match = re.search(r"img(\d+)\.tif", fname)
            if tile_id_match:
                tile_id = tile_id_match.group(1)  
            else:
                tile_id = os.path.splitext(fname)[0]  

            saved_masks[tile_id] = pred

            print(f"[INFO] Saved prediction: {out_path}")

        except Exception as e:
            print(f" Failed {fname}: {e}")
            continue

    print(f" Inference complete. All masks saved in '{out_dir}'")
    return saved_masks




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagepath", type=str, required=True,
                        help="Path to JSON file containing band directories")
    parser.add_argument("--output", type=str, required=False, help="Ignored")
    args = parser.parse_args()

    with open(args.imagepath, "r") as f:
        band_dirs = json.load(f)

    masks = maskgeration(band_dirs, args.output)
    print(f"[INFO] Generated masks for {len(masks)} tiles")
