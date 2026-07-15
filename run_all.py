"""
OmniReview — Master Runner Script
Upload your repository files to Google Drive, mount in Colab, and run:
    %cd /content/drive/MyDrive/OmniReview
    !python run_all.py
This executes all .py files in the correct order (same as "Run All" in the notebook).
"""
import os
import sys
# All files in execution order (mirrors notebook cell sequence)
EXECUTION_ORDER = [
    # 1. Imports, config, constants, utilities
    "models/config.py",
    # 2. Data download, cleaning, splitting
    "data/preprocess.py",
    # 3. EDA (optional — comment out to skip visualizations)
    "data/eda.py",
    # 4. S-BERT embedding extraction
    "data/embed.py",
    # 5. Model definitions
    "models/classifier.py",
    "models/baseline.py",
    "models/vae/vae.py",
    "models/transformer/transformer.py",
    "models/flow/flow.py",
    "models/diffusion/diffusion.py",
    "models/gan/gan.py",
    "models/pipeline.py",
    # 6. Training
    "train/train_cnn.py",
    "train/train_rnn.py",
    "train/train_vae.py",
    "train/train_t5.py",
    "train/train_flow.py",
    "train/train_diffusion.py",
    "train/train_gan.py",
    # 7. Pipeline inference demos
    "experiments/run_pipeline.py",
    # 8. Evaluation and ablation
    "experiments/evaluate_models.py",
    # 9. Appendix diagnostics (optional — comment out to skip)
    "experiments/appendix_diagnostics.py",
]
ROOT = os.path.dirname(os.path.abspath(__file__))
shared_namespace = {"__name__": "__main__", "__file__": __file__}
print("=" * 60)
print("OmniReview -- Full Pipeline Execution")
print("=" * 60)
for filepath in EXECUTION_ORDER:
    full_path = os.path.join(ROOT, filepath)
    if not os.path.exists(full_path):
        print(f"\n  SKIP (not found): {filepath}")
        continue
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {filepath}")
    print(f"{'=' * 60}")
    with open(full_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        exec(compile(code, full_path, 'exec'), shared_namespace)
    except Exception as e:
        print(f"\n  ERROR in {filepath}: {e}")
        print(f"  Stopping execution.")
        sys.exit(1)
print("\n" + "=" * 60)
print("  ALL SCRIPTS COMPLETED SUCCESSFULLY")
print("=" * 60)