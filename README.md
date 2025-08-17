# DCM-SEAL
**Discrete Choice Model with Segmentation & Embedding via Adaptive Learning**

DCM-SEAL is a PyTorch Lightning framework for estimating **latent-class discrete choice models** with **embeddings** and **segmentation networks**. The project includes data preprocessing utilities, efficient data loaders, a configurable model, and an Optuna-driven hyperparameter search pipeline.

---

## Features
- **Flexible latent-class architecture** — shared or class-specific global embedding matrices, segmentation networks, and alternative-specific constants.
- **Efficient dataset handling** — pre-padded `PaddedChoiceDataset` for O(1) item retrieval and default collate compatibility.
- **Automated data preparation** — categorical embeddings, segmentation variables, and train/test splits.
- **End-to-end training runner** — loads data, builds `DataLoader`s, and trains/evaluates the model via PyTorch Lightning.
- **Hyperparameter optimization** — Optuna with dynamic batch size and epoch calculations.

---

## Repository Structure
```text
.
├── src/
│   ├── data_loader.py        # Dataset classes & fast pre-padded dataset
│   ├── data_processing.py    # CSV loading & preprocessing utilities
│   ├── dcm_seal_model.py     # Core PyTorch Lightning model
│   └── run_model.py          # Stand-alone training entry point
├── main.py                   # Optuna hyperparameter search driver
├── genSynthData.py           # Synthetic data generator
├── data/                     # Example datasets (Synthesized, TwinCitiesPath)
└── legacy/                   # Archived scripts
```

---

## Getting Started

### 1) Prepare data
Place `dfIn.csv` (and optionally `dfConv.csv` for categorical conversions) inside:

```
data/<DatasetName>/
```

### 2) Run a single experiment
Edit the config dictionary in `src/run_model.py`, or import and pass a config:

```bash
python src/run_model.py
```

### 3) Hyperparameter search
Set `DATA2USE` in `main.py`, then launch:

```bash
python main.py
```

Artifacts:
- Logs: `logs/`
- Best checkpoints: `checkpoints/`

---

## Data
- `data/Synthesized/dfIn.csv` — toy dataset for quick smoke tests  
- `data/TwinCitiesPath/` — example real-world transit dataset  
- `genSynthData.py` — utility for generating synthetic choice data

---

## Contact
**Kwangho Baek**  
baek0040@umn.edu · dptm22203@gmail.com
