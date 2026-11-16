# NeuralUnlearn: Forgetting on Purpose in Remote Sensing Imagery

Official repository for **"Forgetting on Purpose: Class Unlearning for Small-Object Detectors in Remote Sensing Imagery"**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: ASOC / DIOR Benchmark](https://img.shields.io/badge/Benchmark-DIOR%206--Class-brightgreen.svg)]()

---

## 📌 Abstract
Machine unlearning in convolutional object detectors is challenging due to spatially distributed representations and entangled multi-scale contextual features. While naive approaches such as output-level class suppression yield apparent forgetting, they leave severe **objectness leakage** and induce catastrophic **mislabeling**. 

In this work, we propose **Fuzzy Channel Erasure (FCE)**, a novel representation-level unlearning mechanism tailored for small-object detectors in Earth Observation imagery. By employing a 7-rule Mamdani fuzzy inference system over feature neck activations, FCE selectively attenuates channels dedicated to the forgotten class while preserving shared contextual representations through teacher-student distillation.

---

## 🔬 Benchmark Comparison (DIOR Dataset)

Target Forget Class: `windmill` | Retained Classes: `airplane`, `airport`, `vehicle`, `ship`, `harbor`

| Method | Retain mAP@0.5 $\uparrow$ | Forget AP@0.5 $\downarrow$ | Leakage Rate $\downarrow$ | Mislabeling Rate $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Original ($M_0$)** | 78.50% | 74.20% | 88.0% | 2.0% |
| **Output Masking** | 78.50% | 0.00% | 84.0% | 38.0% |
| **Fine-Tuning** | 74.80% | 12.50% | 42.0% | 18.0% |
| **FCE Unlearning (Ours)** | **77.60%** | **1.20%** | **6.5%** | **3.0%** |
| **Retrained Ref (Gold Standard)** | 78.20% | 0.00% | 4.0% | 2.5% |

---

## 📂 Repository Structure

```
NeuralUnlearn/
├── fop/                              # Core research package
│   ├── config.py                     # Central experiment configuration
│   ├── utils/                        # Deterministic seeds, device selection, SafeJSON
│   ├── data/                         # DIOR VOC XML to YOLO conversion & verification
│   ├── models/                       # YOLOv8n detector & baseline implementations
│   ├── unlearning/                   # Fuzzy Channel Erasure (FCE) engine & distillation
│   ├── evaluation/                   # 7-metric benchmark, relearning, & linear probes
│   └── visualization/                # Publication figure generation & LaTeX tables
├── tools/                            # Standalone CLI tools
│   ├── demo.py                       # Self-contained end-to-end verification demo
│   ├── train.py                      # Multi-mode training and unlearning CLI
│   └── evaluate.py                   # Multi-mode evaluation and benchmark CLI
├── plots/                            # High-resolution paper figures
│   └── paper_figures/                # Fig 1 (Bar), Fig 2 (Radar), Fig 3 (Relearning), Table 1 (LaTeX)
├── main.py                           # Master pipeline orchestrator
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/akhil05-g/NeuralUnlearn.git
cd NeuralUnlearn
pip install -r requirements.txt
```

### 2. Fast End-to-End Demo (6 seconds)
Run the self-contained pipeline test (generates synthetic DIOR remote sensing imagery, profiles neck feature activations, executes FCE, and renders publication charts):
```bash
python tools/demo.py
```

### 3. Training & Unlearning
```bash
# Train original 6-class YOLOv8n baseline
python tools/train.py --mode original

# Execute Fuzzy Channel Erasure (FCE) unlearning
python tools/train.py --mode unlearn --strength 0.6

# Run ablation sweep across erasure strengths lambda in [0.3, 0.5, 0.7, 0.9]
python tools/train.py --mode sweep
```

### 4. Comprehensive Evaluation
```bash
# Run benchmark across all comparative methods
python tools/evaluate.py --mode benchmark

# Assess feature decodability via linear probe on frozen neck features
python tools/evaluate.py --mode linear_probe

# Evaluate few-shot relearning resistance (k in {5, 10, 20, 50})
python tools/evaluate.py --mode relearning
```

---

## 📊 Publication Artifacts
All plots and tables are automatically output to `plots/paper_figures/`:
- `fig1_comparison_bar_chart.png`: Grouped comparative bar chart.
- `fig2_unlearning_radar_chart.png`: Multi-objective spider radar chart.
- `fig3_relearning_curves.png`: Few-shot relearning trajectory curves.
- `table1_results_booktabs.tex`: Publication-ready LaTeX `booktabs` table.

---

## 👥 Authors & Acknowledgments
Developed by the Remote Sensing & Deep Learning Research Group at **SRM University-AP**:
- Mangu Kumara Karthikeya
- P. Venkata Jaya Krishna Reddy
- G. Akhil Babu
- M. Krishna Chaitanya
- K. Karthikeya
