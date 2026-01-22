import numpy as np
import pandas as pd
import base64
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" 

def decode_patch(patch_str):
    patch_bytes = base64.b64decode(patch_str)
    patch = np.frombuffer(patch_bytes, dtype=np.float32)
    return patch.reshape(15, 35, 35)

def get_stats(patch_section):
    """Helper to get Mean, Std for a specific section of the image."""
    mean_val = np.mean(patch_section, axis=(1, 2))
    std_val = np.std(patch_section, axis=(1, 2))
    return np.concatenate([mean_val, std_val])

def extract_features(patch):
    """
    Extracts Global Stats + Quadrant Stats + Indices.
    Now the model knows WHERE things are (Top-Left vs Center, etc.)
    """
    # 1. Global Indices (NDVI, NDWI) - Best calculated on the whole image
    # Bands: 2=Green, 3=Red, 7=NIR
    green = patch[2]
    red = patch[3]
    nir = patch[7]
    epsilon = 1e-8
    
    ndvi = np.mean((nir - red) / (nir + red + epsilon))
    ndwi = np.mean((green - nir) / (green + nir + epsilon))
    gndvi = np.mean((nir - green) / (nir + green + epsilon))
    
    indices = [ndvi, ndwi, gndvi]

    # 2. Global Stats (The old way)
    global_stats = get_stats(patch)
    
    # 3. Quadrant Stats (The NEW Spatial way)
    # Image is 35x35. Midpoint is roughly 17.
    # We slice the array: patch[:, Y_start:Y_end, X_start:X_end]
    mid = 17
    
    # Top-Left
    q1 = patch[:, :mid, :mid]
    stats_q1 = get_stats(q1)
    
    # Top-Right
    q2 = patch[:, :mid, mid:]
    stats_q2 = get_stats(q2)
    
    # Bottom-Left
    q3 = patch[:, mid:, :mid]
    stats_q3 = get_stats(q3)
    
    # Bottom-Right
    q4 = patch[:, mid:, mid:]
    stats_q4 = get_stats(q4)
    
    # 4. Combine ALL features
    # This creates a massive feature vector (Global + 4 Quadrants + Indices)
    return np.concatenate([
        global_stats, 
        stats_q1, stats_q2, stats_q3, stats_q4, 
        indices
    ])

def load_data():
    """Smart loader (Same as before)"""
    csv_path = DATA_DIR / "train.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "train_labels.csv"
        
    df = pd.read_csv(csv_path)
    
    # Auto-detect folder
    # (Assuming you fixed the folder structure as discussed previously)
    # If using the 'smart search' version, paste that load_data function here.
    # For simplicity, here is the standard dual-folder search:
    
    possible_folders = [DATA_DIR / "patches_part1", DATA_DIR / "patches_part2", DATA_DIR / "train" / "patches_part1"]
    
    X = []
    y = []
    
    print("Extracting SPATIAL features...")
    
    # Find where images are hiding
    valid_folders = [p for p in possible_folders if p.exists()]
    if not valid_folders:
         # Fallback search
         valid_folders = list(DATA_DIR.rglob("*.npy"))
         if valid_folders: valid_folders = [valid_folders[0].parent]
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row['sample_id'] # Check your CSV column name!
        label = row['vistgerd_idx']
        
        found = False
        for folder in valid_folders:
            img_path = folder / f"{file_id}.npy"
            if img_path.exists():
                try:
                    patch = np.load(img_path)
                    X.append(extract_features(patch))
                    y.append(label)
                    found = True
                    break
                except: pass
        
    return np.array(X), np.array(y)