# MAI-T1D-Foundation-Stack

**Stacked multimodal foundation model ensemble for Type 1 Diabetes (NIH MAI-T1D aligned)**  
WGS + RNA-Seq + Clinical metadata → calibrated risk/progression predictions

Forked & heavily extended from [keninayoung/QISICGM](https://github.com/keninayoung/QISICGM) (MIT License) under the MAI-T1D project (2026).

---

## Overview

This repository implements a **production-ready stacked ensemble** of modality-specific foundation models for Type 1 Diabetes:

- **WGS expert** → pre-trained genomic foundation (DNABERT-2, Enformer, etc.)
- **RNA-Seq expert** → pre-trained transcriptomic foundation (Geneformer, scGPT, etc.)
- **Clinical expert** → tabular foundation (TabPFN, FT-Transformer, etc.)

Each expert is wrapped in a lightweight adapter that extracts calibrated probabilities + rich embeddings. A meta-learner then fuses them (late fusion / stacked generalization) for final T1D risk, progression stage, or subtype predictions.

Key features:
- Full out-of-fold (OOF) stacking with per-expert calibration (isotonic + Platt)
- Native support for NVIDIA H200 (bf16/FP8 acceleration via Accelerate)
- Handles missing modalities gracefully
- Extremely fast inference once models are loaded
- Clinical-grade reporting (risk bands, SHAP per modality, reliability plots)

---

## Repository Structure

````
MAI-T1D-Foundation-Stack/
├─ data/                  # sample or placeholder data
├─ models/                # saved meta-learner, calibrators, adapters config
├─ plots/                 # auto-generated evaluation plots
├─ foundation_adapter.py  # core adapter class for any foundation model
├─ stacked_ensemble.py    # T1D-specific stacked foundation model
├─ train_stack.py         # training/fine-tuning script
├─ make_demo_predictions.py # fast inference on new patients
├─ plots_and_reporting.py # all visualization utilities
├─ requirements_h200.txt  # H200-optimized requirements
└─ README.md
````

---

## Requirements (NVIDIA H200)

- Python 3.10+
- CUDA 12.6+ (H200 native)
- PyTorch 2.5+ with Hopper support

---
Installation

```bash
pip install -r requirements_h200.txt


## License

MIT License

Copyright (c) 2026 Kenneth Young, PhD and Dena Tewey, MPH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

git clone https://github.com/USF-HII/MAI-T1D-Foundation-Stack.git
cd MAI-T1D-Foundation-Stack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_h200.txt

````
