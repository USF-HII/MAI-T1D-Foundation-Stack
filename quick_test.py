# quick_test.py
"""
MAI-T1D-Foundation-Stack Quick Test
Run from VS Code (F5 or right-click -> Run Python File) 
or from command line: python quick_test.py

Works on CPU, local GPU, or H200 automatically.
"""

import argparse
import torch
import numpy as np
from stacked_ensemble import T1DStackedFoundationModel

def main():
    parser = argparse.ArgumentParser(description="Quick test for MAI-T1D stacked foundation model")
    parser.add_argument("--wgs_path", type=str, default=None, help="Path to WGS foundation model")
    parser.add_argument("--rnaseq_path", type=str, default=None, help="Path to RNA-Seq foundation model")
    parser.add_argument("--clinical_path", type=str, default=None, help="Path to clinical foundation model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    parser.add_argument("--low_memory", action="store_true", help="Use low-memory mode (for laptops/small GPUs)")
    parser.add_argument("--dummy", action="store_true", help="Run with dummy data (no real models needed)")
    args = parser.parse_args()

    print("MAI-T1D-Foundation-Stack Quick Test")
    print(f"Device: {args.device} | Low memory: {args.low_memory}\n")

    if args.dummy:
        print("Using dummy data (no real models loaded)...")
        # Dummy adapters with minimal mocks for testing
        stacked = T1DStackedFoundationModel(
            device=args.device,
            low_memory=args.low_memory
        )
        # Manually add dummy adapters that return fixed outputs for testing
        class DummyAdapter:
            def predict_and_extract(self, batch):
                return np.array([[0.3]]), None, np.random.rand(1, 512).astype(np.float32)
        
        stacked.adapters = {
            'wgs': DummyAdapter(),
            'rnaseq': DummyAdapter(),
            'clinical': DummyAdapter()
        }
        stacked.meta_learner = None  # will be skipped for dummy
    else:
        stacked = T1DStackedFoundationModel(
            wgs_path=args.wgs_path,
            rnaseq_path=args.rnaseq_path,
            clinical_path=args.clinical_path,
            device=args.device,
            low_memory=args.low_memory
        )

    # Create dummy patient input (shape doesn't have to be perfect for quick test)
    sample_data = {
        'wgs': torch.randn(1, 512).to("cuda" if torch.cuda.is_available() else "cpu"),
        'rnaseq': {"input_ids": torch.randint(0, 30000, (1, 512))},
        'clinical': np.random.rand(1, 20).astype(np.float32)
    }

    print("Running inference on dummy patient...")
    try:
        prob = stacked.predict(sample_data)
        print(f"\n SUCCESS!")
        print(f"T1D risk probability: {prob:.4f} ({prob*100:.1f}%)")
        print(f"Risk band: {'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'}")
    except Exception as e:
        print(f"\n Error during prediction: {e}")
        print("Tip: Use --dummy flag first to test the pipeline without real models.")

    print("\n Quick test complete! You can now:")
    print("   1. Replace dummy paths with real foundation model paths")
    print("   2. Run full training with train_stack.py (coming next)")
    print("   3. Run in VS Code: just press F5 or Ctrl+F5")

if __name__ == "__main__":
    main()
