# NeuroSys

NeuroSys is a research-grade AIOps framework for modeling distributed system behavior from logs using probabilistic latent representations with Variational Autoencoders (VAE).

## Core Capabilities

- Log parsing and sequence construction (HDFS Xu style block-centric sequences)
- Baseline anomaly detection (Isolation Forest, PCA reconstruction)
- VAE-based latent behavior modeling and anomaly scoring
- Hybrid temporal detector (Temporal VAE + PCA ensemble)
- Latent trajectory and velocity-based failure prediction
- Unsupervised failure typing via clustering
- Counterfactual root cause reasoning and causal approximation
- Cross-system generalization support (e.g., BGL, OpenStack)
- Synthetic anomaly injection for controlled evaluation
- FastAPI interfaces for ingestion, detection, and explanations

## Project Layout

- `src/preprocessing`: parsing, sequence construction, dataset adapters
- `src/features`: bag-of-events feature extraction and normalization
- `src/models`: baseline and VAE models
- `src/analysis`: latent visualization, trajectories, typing, explanations
- `src/evaluation`: metrics and plots
- `src/api`: service API definitions
- `experiments`: reproducible experiment entrypoints
- `configs`: experiment configurations
- `results`: saved metrics and visual artifacts

## Quick Start

1. Create and use local virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install in the venv:
   - `python -m pip install -e .`
3. Place HDFS Xu dataset files in `data/` (see supported formats below).
4. Run experiment:
   - `python experiments/run_pipeline.py --config configs/default.yaml`
   - `python experiments/run_pipeline.py --config configs/full_scale.yaml`
   - `python experiments/run_pipeline.py --config configs/temporal_smoke.yaml`
   - `python experiments/run_pipeline.py --config configs/temporal_full_scale.yaml`
   - `python experiments/run_pipeline.py --config configs/temporal_hybrid.yaml`
5. Start API:
   - `python -m api.server`

## Latest Benchmark Snapshot

From the latest temporal hybrid run (`results/temporal_hybrid_run/summary.json`):

- Temporal VAE + PCA ensemble: F1 `0.9133`, ROC-AUC `0.9974`
- PCA reconstruction baseline: F1 `0.5138`, ROC-AUC `0.9967`
- Isolation Forest baseline: F1 `0.4425`, ROC-AUC `0.8821`

This means the hybrid ensemble currently outperforms standalone PCA on F1 by a large margin on the HDFS Xu split setup.

## Expected HDFS Xu Files

- `data/hdfs_xu/HDFS.log` (raw logs)
- `data/hdfs_xu/anomaly_label.csv` with columns similar to `BlockId,Label`
- Optional: `data/hdfs_xu/HDFS.log_structured.csv` with `EventId` already parsed

## Supported HDFS Xu Split Format

The current default config also supports pre-grouped sequence splits directly:

- `data/hdfs_train.txt`
- `data/hdfs_test_normal.txt`
- `data/hdfs_test_abnormal.txt`

Each line format:

- `<BlockId>,<space-separated event IDs>`

The bag-of-events vocabulary is fit on training sequences only, and unseen test events are mapped to an explicit `<UNK>` bucket.
