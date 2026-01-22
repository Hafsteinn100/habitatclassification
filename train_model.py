import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from utils import extract_features

# ==========================================
# FILE CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent / "data"
PART1_PATH = BASE_DIR / "train" / "patches_part1.npy"
PART2_PATH = BASE_DIR / "train" / "patches_part2.npy"
LABELS_PATH = BASE_DIR / "train.csv"
# ==========================================

def train():
    print("--- STARTING HEAVYWEIGHT CHAMPION TRAINING ---")
    
    # 1. Load Data
    print("Loading image data...")
    try:
        part1 = np.load(PART1_PATH)
        part2 = np.load(PART2_PATH)
        patches = np.concatenate([part1, part2], axis=0)
    except FileNotFoundError:
        print("[ERROR] Data files not found.")
        return

    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_PATH)
    labels = labels_df["vistgerd_idx"].values

    # 2. Extract Features
    print("Extracting features (Make sure utils.py has the NDVI/NDWI code!)...")
    X = np.array([extract_features(p) for p in patches])
    y = labels
    
    # 3. Define the 3 Experts
    print("Initializing the Committee...")
    
    # Expert 1: Random Forest (Classic reliability)
    # Increased to 500 trees and deeper depth since size doesn't matter
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,        # Deeper logic
        n_jobs=-1,
        random_state=42
    )
    
    # Expert 2: Extra Trees (The "Wild Card")
    # Often beats Random Forest on texture-heavy data like moss/lava
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        bootstrap=False,     # Uses raw data (better for texture)
        n_jobs=-1,
        random_state=42
    )
    
    # Expert 3: Gradient Boosting (Precision)
    gb = HistGradientBoostingClassifier(
        max_iter=100,        # Learn longer
        learning_rate=0.05,  # Learn slower/more carefully
        max_depth=12,
        l2_regularization=1.0,
        random_state=42
    )
    
    # 4. Create the Voting Block
    # They vote, and we take the most confident answer (Soft Voting)
    print("Stacking models into VotingClassifier...")
    voting_model = VotingClassifier(
        estimators=[
            ('rf', rf), 
            ('et', et), 
            ('gb', gb)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # 5. Train
    print("Training on 100% of data (This might take 2-3 minutes)...")
    voting_model.fit(X, y)
    
    # 6. Save
    output_path = Path(__file__).parent / "model.joblib"
    print(f"Saving heavy model to {output_path}...")
    
    # compress=3 keeps it loadable but doesn't crush it too small
    joblib.dump(voting_model, output_path, compress=3)
    print("DONE! This is your strongest possible model.")

if __name__ == "__main__":
    train()