import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from utils import load_data

def train():
    print("--- STARTING MAXIMUM QUALITY TRAINING (Spatial 3x3) ---")
    
    # 1. Load Data
    # The new utils.py automatically handles the 3x3 grid feature extraction
    print("Loading data and extracting Spatial Pyramid features (330 features)...")
    X, y = load_data()
    
    print(f"Dataset Shape: {X.shape}")
    
    # 2. Define the Experts (Optimized for Weighted F1 Score)
    
    # Expert 1: Random Forest (The Generalist)
    # increased to 1000 trees for stability
    # min_samples_leaf=2 prevents overfitting on single noisy pixels
    print("Initializing Random Forest (1000 trees)...")
    rf = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=None,      # Allow full depth
        min_samples_leaf=2,  # Slight regularization (Crucial for test set score)
        n_jobs=-1,
        random_state=42
    )
    
    # Expert 2: Extra Trees (The Texture Specialist)
    # UPGRADE: Increased from 100 -> 1000 trees. 
    # ExtraTrees needs high estimator counts to beat Random Forest.
    print("Initializing Extra Trees (1000 trees)...")
    et = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=-1,
        random_state=42
    )
    
    # Expert 3: Gradient Boosting (The Mathematician)
    # UPGRADE: More iterations (1000) with slower learning rate (0.03)
    # This finds subtle patterns that the other two miss.
    print("Initializing Gradient Boosting (Deep Learning mode)...")
    gb = HistGradientBoostingClassifier(
        max_iter=1000,       # Double the thinking time
        learning_rate=0.03,  # Learn slower = higher precision
        max_depth=15,        # Allow deeper logic for complex moss/lava borders
        l2_regularization=0.5,
        random_state=42
    )
    
    # 3. Voting
    # We use soft voting (averaging probabilities)
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
    
    # 4. Train
    print("Training on full dataset (This will take 2-5 minutes)...")
    voting_model.fit(X, y)
    
    # 5. Save
    output_path = Path(__file__).parent / "model.joblib"
    print(f"Saving model to {output_path}...")
    
    # Compress=3 balances save speed and file size
    joblib.dump(voting_model, output_path, compress=3)
    print("DONE! Your strongest model is ready.")

if __name__ == "__main__":
    train()