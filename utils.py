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
    Extracts advanced satellite features with Spatial Awareness.
    
    New Strategy: "Spatial Pyramid"
    1. Global Stats (Whole 35x35)
    2. Grid Stats (Split into 3x3 grid of approx 11x11 pixels)
    
    This allows the model to differentiate "moss in center" vs "moss at edge".
    """
    
    # --- Helper to calculate stats for a block ---
    def get_pixel_stats(pixels):
        # pixels shape: (15, N_pixels)
        # We want mean and std for each band (30 features)
        mean = np.mean(pixels, axis=1) # Shape (15,)
        std = np.std(pixels, axis=1)   # Shape (15,)
        
        # Spectral Indices Stats (Mean ONLY to save dims)
        green = pixels[2]
        red = pixels[3]
        nir = pixels[7]
        epsilon = 1e-8
        
        # Calculate indices per pixel then take mean
        ndvi_mean = np.mean((nir - red) / (nir + red + epsilon))
        ndwi_mean = np.mean((green - nir) / (green + nir + epsilon))
        gndvi_mean = np.mean((nir - green) / (nir + green + epsilon))
        
        return np.concatenate([
            mean, 
            std, 
            [ndvi_mean, ndwi_mean, gndvi_mean]
        ])

    # 1. Global Features (The original strategy)
    # Flatten spatial dims: (15, 35, 35) -> (15, 1225)
    global_pixels = patch.reshape(15, -1)
    global_feats = get_pixel_stats(global_pixels)
    
    # 2. Grid Features (3x3 Split)
    # 35 pixels / 3 is approx 11. 
    # Slices: 0-11, 11-23, 23-35
    h_slices = [slice(0, 11), slice(11, 23), slice(23, 35)]
    w_slices = [slice(0, 11), slice(11, 23), slice(23, 35)]
    
    grid_feats = []
    
    for hs in h_slices:
        for ws in w_slices:
            # Extract sub-patch
            sub_patch = patch[:, hs, ws]
            # Flatten
            sub_pixels = sub_patch.reshape(15, -1)
            # Calc stats
            feats = get_pixel_stats(sub_pixels)
            grid_feats.append(feats)
            
    # Concatenate everything
    # Global (33 feats) + 9 * Grid (33 feats) = 330 features
    return np.concatenate([global_feats] + grid_feats)

def load_data():
    """Simple and robust loader."""
    print("Loading data from standard paths...")
    
    # Paths
    base_dir = DATA_DIR
    part1_path = base_dir / "train" / "patches_part1.npy"
    part2_path = base_dir / "train" / "patches_part2.npy"
    labels_path = base_dir / "train.csv"
    
    # Load Labels
    df = pd.read_csv(labels_path)
    labels = df["vistgerd_idx"].values
    
    # Load Patches
    # Check if files exist
    if not part1_path.exists():
         # Try fallback (maybe they are directly in data?)
         part1_path = base_dir / "patches_part1.npy"
         part2_path = base_dir / "patches_part2.npy"
         
    p1 = np.load(part1_path)
    p2 = np.load(part2_path)
    patches = np.concatenate([p1, p2], axis=0)
    
    print(f"Patches loaded: {patches.shape}")
    
    # Extract features
    X = []
    print("Extracting features (This is fast)...")
    for p in tqdm(patches):
        X.append(extract_features(p))
        
    return np.array(X), labels