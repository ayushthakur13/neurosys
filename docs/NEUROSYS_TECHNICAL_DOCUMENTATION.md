# NeuroSys Technical Documentation

## 1. Project Purpose

NeuroSys is a research-grade AIOps framework for modeling distributed system behavior from logs using probabilistic latent representations learned by a Variational Autoencoder (VAE). The implemented system supports:

- Log/sequence preprocessing
- Feature engineering (bag-of-events)
- Baseline anomaly detection (Isolation Forest, PCA reconstruction)
- VAE-based anomaly detection
- Latent-space analysis
- Trajectory and latent-velocity analysis
- Failure typing via clustering
- Counterfactual root-cause analysis
- Approximate causal graph construction
- Cross-system evaluation hooks
- Reproducible experiment outputs
- API interfaces for detection, latent retrieval, and root-cause explanation

## 2. What Was Implemented In This Workspace

### 2.1 Repository Initialization

Created the modular project structure:

- configs
- data
- experiments
- notebooks
- results
- src
- tests
- docs

Initial scaffold mistakenly created under a nested directory and then moved to repository root.

### 2.2 Packaging and Dependencies

Created project metadata in pyproject.toml with:

- Build system: setuptools
- Package root: src
- Runtime dependencies: numpy, pandas, scikit-learn, torch, matplotlib, seaborn, networkx, pyyaml, fastapi, uvicorn, umap-learn
- Dev dependencies: pytest, ruff

### 2.3 Configurations

Implemented:

- configs/default.yaml: main experiment configuration
- configs/smoke.yaml: lightweight validation config
- configs/full_scale.yaml: full-scale experiment configuration
- configs/temporal_smoke.yaml: temporal sequence-model smoke config
- configs/temporal_full_scale.yaml: temporal sequence-model full-scale config
- configs/temporal_hybrid.yaml: optimized temporal config with KL-weighted scoring and VAE+PCA ensemble reporting

The default config now supports your current HDFS Xu split-file format under data/.

### 2.4 Dataset Integration

Detected and integrated these files:

- data/hdfs_train.txt
- data/hdfs_test_normal.txt
- data/hdfs_test_abnormal.txt

Each line format:

- Block ID, followed by a space-separated event-ID sequence
- Example pattern: blk_xxx,5 22 5 11 9 ...

These are treated as pre-grouped execution instances (one sequence per block ID).

## 3. Module-Level Implementation Details

## 3.1 Utility Modules

### src/utils/io.py

- ensure_dir(path): creates output directories recursively
- write_json(data, path): writes structured JSON artifacts
- read_json(path): reads JSON artifacts

### src/utils/repro.py

- set_seed(seed): controls Python, NumPy, and Torch seeds
- enables deterministic CuDNN flags

### src/utils/config.py

- load_yaml(path): loads YAML experiment configuration

## 3.2 Preprocessing

### src/preprocessing/drain_parser.py

Implements a lightweight Drain-like parser for raw logs:

- Token normalization for numbers, hex-like tokens, IP-like tokens, block IDs, and long mixed tokens
- Clustering templates by token length
- Similarity matching with wildcard updates
- Event template to EventId mapping (E1, E2, ...)

Main classes:

- ParsedLog dataclass
- SimpleDrainParser

### src/preprocessing/hdfs.py

Implements raw/structured HDFS preprocessing:

- HDFSPreprocessor extracts block IDs via regex (blk_-?\d+)
- Supports:
  - raw logs + labels
  - structured logs with EventId
- Groups by BlockId in line order to produce sequences
- Maps labels to binary anomaly labels

Data structure:

- SequenceDataset dataclass:
  - block_ids
  - sequences
  - labels
  - templates

### src/preprocessing/sequence_splits.py

Implements direct loading for your hdfs_xu split files:

- HDFSXuSplitPreprocessor
- Reads train/normal-test/abnormal-test split files
- Produces SequenceDataset with labels:
  - train -> 0
  - test normal -> 0
  - test abnormal -> 1

### src/preprocessing/synthetic.py

Implements synthetic anomaly injection:

- Modes: insert, remove, shuffle
- Applies perturbations to a sampled subset of normal sequences
- Returns modified labels and injected indices

Classes:

- SyntheticInjectionResult
- SyntheticInjector

## 3.3 Feature Engineering

### src/features/bag_of_events.py

Implements bag-of-events representation:

- Builds event vocabulary with min_count threshold
- Fits vocabulary on training sequences only in the strict pipeline
- Uses an explicit <UNK> bucket for unseen events
- Converts each sequence to frequency vector over event IDs
- Optional normalization using sklearn normalize (default l2)

Objects:

- BoEResult dataclass
- BagOfEventsVectorizer

### src/features/temporal_vectorizer.py

Implements sequence-aware representation:

- Preserves event order
- Produces padded token IDs
- Produces position indices and masks
- Supports max sequence length and unknown-token handling

Objects:

- TemporalBatch dataclass
- SequenceAwareVectorizer

## 3.4 Baseline Models

### src/models/baselines.py

Implemented baseline anomaly scoring:

- IsolationForest trained on normal-only data
- PCA reconstruction error trained on normal-only data

Methods:

- fit_normal(X, y)
- score(X) -> isolation_forest scores, pca reconstruction scores

Objects:

- BaselineScores dataclass
- BaselineRunner

## 3.5 VAE Model

### src/models/vae.py

Implemented a feedforward VAE with:

- Encoder MLP -> mu/logvar
- Reparameterization trick
- Decoder MLP with sigmoid output
- ELBO-style objective:
  - reconstruction MSE
  - KL divergence scaled by beta

Training setup:

- Normal-only training data
- Adam optimizer
- Batch DataLoader

Inference methods:

- reconstruction_error(X)
- latent(X) (uses encoder mean)

Objects:

- VAEConfig dataclass
- VAE nn.Module
- VAETrainer

### src/models/temporal_vae.py

Implements the temporal VAE path:

- Bidirectional GRU encoder and GRU decoder
- Dropout-regularized embeddings and encoder normalization
- Mask-aware reconstruction loss
- Validation-based beta-optimized training path
- Latent extraction on ordered sequences
- KL-weighted anomaly scoring (`reconstruction + kl_weight * KL`)

Objects:

- TemporalVAEConfig dataclass
- TemporalVAE nn.Module
- TemporalVAETrainer

## 3.6 Evaluation

### src/evaluation/metrics.py

Implemented:

- threshold_by_percentile(scores, p)
- compute_binary_metrics(y_true, scores, threshold)

Metrics computed:

- precision
- recall
- F1
- ROC-AUC
- TN, FP, FN, TP

### src/evaluation/plots.py

Implemented plot generation for:

- ROC curve
- confusion matrix heatmap
- 2D latent scatter

## 3.7 Analysis Modules

### src/analysis/latent.py

Latent projection methods:

- PCA (fast option, added for smoke runs)
- t-SNE
- UMAP with fallback to t-SNE

### src/analysis/trajectory.py

Implemented:

- build_trajectories: groups latent vectors per block ID
- latent_velocity: first-order difference over ordered latent states
- trajectory_risk_score: cosine similarity between current and failure velocity means

### src/analysis/failure_typing.py

Implemented anomaly clustering:

- DBSCAN or KMeans over anomalous latent vectors
- cluster summary with top dominant events and interpretable type labels

### src/analysis/counterfactual.py

Implemented greedy counterfactual explanation on bag-of-events vector:

- Iteratively removes high-frequency events
- Accepts changes that reduce anomaly score
- Limits number of changes
- Returns event-level explanation steps

Also includes dominant_event_drift helper.

### src/analysis/causal.py

Implemented approximate causal structure:

- Temporal precedence counts for event transitions
- Anomaly-conditional edge bias score
- Directed graph with edge attributes:
  - anomaly_bias
  - anomaly_count
  - total_count

Includes graph_to_dict serialization helper.

## 3.8 API Layer

### src/api/schemas.py

Pydantic request/response schemas:

- SequenceInput
- DetectResponse
- LatentResponse
- RootCauseResponse

### src/api/service.py

NeuroSysService:

- Loads artifacts (vocab, VAE checkpoint, scores)
- Sequence -> feature conversion
- detect(events) -> score + anomaly flag
- latent(events) -> latent vector
- root_cause(events) -> counterfactual explanation

### src/api/server.py

FastAPI endpoints:

- GET /health
- POST /detect
- POST /latent
- POST /root-cause

Startup behavior:

- Tries loading artifacts from results/default_run/artifacts
- API still starts if artifacts are not present yet

## 4. Experiment Runner

### experiments/run_pipeline.py

This is the main orchestrator.

Pipeline steps:

1. Load config
2. Set seed
3. Prepare output directories
4. Load primary dataset:
   - pregrouped split mode (your files) or
   - raw/structured HDFS mode
5. Split-aware training setup:
  - fit bag-of-events vocabulary on training normal sequences only
  - keep an explicit unknown bucket for unseen test events
  - train baselines and VAE on training normal data only
6. Optionally inject synthetic anomalies into the training split
7. Build bag-of-events features for train and evaluation splits
8. Train baselines on normal training data
9. Score the held-out evaluation split and evaluate baselines
10. Hyperparameter search for VAE over:
   - latent_dims
   - beta values
11. Select best VAE by F1 score on evaluation split
12. Extract latent representations for evaluation sequences
13. Project latent to 2D
14. Cluster anomalous latents for failure typing
15. Compute trajectories and global risk score
16. Generate counterfactual explanations for anomaly samples
17. Build approximate causal graph
18. Optionally evaluate on second dataset (cross-system)
19. Save artifacts, plots, and summary

For temporal runs, the pipeline also reports an ensemble detector:

- temporal_vae_pca_ensemble = average of best temporal VAE anomaly scores and PCA reconstruction scores
- dedicated ROC/confusion-matrix plots and summary metrics for this ensemble

Saved artifacts:

- X.npy
- X_train.npy
- X_eval.npy
- y.npy
- y_train.npy
- latent.npy
- scores_vae.npy
- vocab.json
- vae.pt

Saved plots:

- ROC for each model
- confusion matrix for each model
- latent space scatter

Saved report:

- summary.json (full experiment output)

## 5. Environment and Dependency Handling

## 5.1 Global Environment Constraint

A global editable install failed due externally managed Python environment policy (PEP 668). To comply with your requirement, all subsequent installs and executions were moved into local virtual environment.

## 5.2 Virtual Environment Used

Local interpreter used:

- /Users/ayushpratapsingh/dev/Projects/neurosys/.venv/bin/python

Package install performed inside .venv:

- pip install -e .
- runtime packages installed in .venv site-packages

No further dependency installation was performed in global environment.

## 6. Validation and Runs

## 6.1 Smoke Run

Executed:

- python experiments/run_pipeline.py --config configs/smoke.yaml

Result:

- Completed successfully
- Output summary generated at results/smoke_run/summary.json

Smoke run size details (from summary):

- num_sequences: 575061
- num_anomalies: 16838

Smoke run key metrics:

- Isolation Forest: F1 0.3634, ROC-AUC 0.9326
- PCA Reconstruction: F1 0.4349, ROC-AUC 0.9751
- VAE best (z=8, beta=1.0, 2 epochs): F1 0.2725, ROC-AUC 0.7958

## 6.2 Full Default Run Note

The default config is significantly heavier:

- more VAE hyperparameter combinations
- more epochs
- UMAP embedding by default

During manual iterative checks, long runs were interrupted interactively before completion; smoke configuration was introduced to confirm end-to-end correctness quickly.

## 6.3 Full-Scale Run

Executed:

- .venv/bin/python experiments/run_pipeline.py --config configs/full_scale.yaml

Result:

- Completed successfully
- Summary generated at results/full_scale_run/summary.json

Full-scale run highlights:

- num_sequences_train: 5583
- num_sequences_eval: 569478
- num_anomalies_train: 0
- num_anomalies_eval: 16838
- VAE grid: 9 runs (latent_dims [8, 16, 32] x beta [0.5, 1.0, 2.0])
- best VAE: latent_dim 16, beta 1.0, epochs 15

Full-scale key metrics:

- Isolation Forest: F1 0.3561, ROC-AUC 0.9267
- PCA Reconstruction: F1 0.4338, ROC-AUC 0.9752
- VAE best: F1 0.2729, ROC-AUC 0.7814

## 6.4 Temporal Full-Scale Run

Executed:

- .venv/bin/python experiments/run_pipeline.py --config configs/temporal_full_scale.yaml

Result:

- Completed successfully
- Summary generated at results/temporal_full_scale_run/summary.json

Highlights:

- num_sequences_train: 4466
- num_sequences_eval: 569478
- num_anomalies_eval: 16838

Key metrics:

- Isolation Forest: F1 0.1535, ROC-AUC 0.8772
- PCA Reconstruction: F1 0.5138, ROC-AUC 0.9962
- Temporal VAE best: F1 0.0577, ROC-AUC 0.9886

Observation:

- Standalone temporal VAE had strong ranking quality (high ROC-AUC) but poor thresholded precision/recall balance in this configuration.

## 6.5 Temporal Hybrid Run (Improved)

Executed:

- .venv/bin/python experiments/run_pipeline.py --config configs/temporal_hybrid.yaml

Result:

- Completed successfully
- Summary generated at results/temporal_hybrid_run/summary.json

Configuration changes vs temporal_full_scale:

- lower learning rate (0.0005)
- validation split 0.3
- KL-weighted temporal scoring enabled
- temporal VAE + PCA ensemble reporting enabled

Key metrics:

- Isolation Forest: F1 0.4425, ROC-AUC 0.8821
- PCA Reconstruction: F1 0.5138, ROC-AUC 0.9967
- Temporal VAE best: F1 0.0576, ROC-AUC 0.9953
- Temporal VAE + PCA Ensemble: F1 0.9133, ROC-AUC 0.9974

Conclusion:

- The hybrid ensemble is currently the best-performing detector in this repository.

## 7. Current Configuration Details

## 7.1 Default Config Highlights (configs/default.yaml)

- dataset format: pregrouped_sequences
- data root: data
- files:
  - hdfs_train.txt
  - hdfs_test_normal.txt
  - hdfs_test_abnormal.txt
- baselines enabled
- VAE search over latent_dims and beta
- synthetic injection enabled
- embedding method: umap

## 7.2 Smoke Config Highlights (configs/smoke.yaml)

- reduced baseline estimator count
- VAE epochs lowered to 2
- single latent dimension and beta
- synthetic injection disabled
- embedding method set to pca for speed

## 8. Reproducibility and Output Contracts

Reproducibility mechanisms:

- deterministic seed setting in utils/repro.py
- YAML-driven configuration
- structured output directories under results/<run_name>
- model checkpoint and vocabulary persistence
- summary JSON with metrics and analysis outputs

Output contract under each run directory:

- artifacts/
- plots/
- summary.json

## 9. API Usage Contract

Expected sequence payload format:

- block_id: string
- events: list of event IDs as strings

Detect response:

- block_id
- score
- is_anomaly

Latent response:

- block_id
- latent vector

Root cause response:

- block_id
- explanation object with counterfactual edits

## 10. Known Technical Limitations

1. Trajectory modeling currently uses one latent point per block sequence in this dataset mode, so longitudinal trajectory depth is limited unless time-windowed/streamed instances are introduced.
2. Counterfactual routine is greedy and feature-space based (bag-of-events); it is practical but not globally optimal.
3. Causal graph is statistical temporal approximation, not formal causal identification.
4. Cross-system generalization path is implemented but requires second dataset files to be present and configured.
5. API startup artifact path defaults to results/default_run/artifacts; if only smoke_run exists, either copy artifacts or update service path.
6. Standalone temporal VAE may still underperform under hard thresholds; operational use should prefer the temporal ensemble output where available.

## 11. File Index (Implemented Core)

Top-level:

- pyproject.toml
- README.md
- configs/default.yaml
- configs/smoke.yaml
- experiments/run_pipeline.py

Package:

- src/__init__.py
- src/utils/config.py
- src/utils/io.py
- src/utils/repro.py
- src/preprocessing/drain_parser.py
- src/preprocessing/hdfs.py
- src/preprocessing/sequence_splits.py
- src/preprocessing/synthetic.py
- src/features/bag_of_events.py
- src/models/baselines.py
- src/models/vae.py
- src/evaluation/metrics.py
- src/evaluation/plots.py
- src/analysis/latent.py
- src/analysis/trajectory.py
- src/analysis/failure_typing.py
- src/analysis/counterfactual.py
- src/analysis/causal.py
- src/api/schemas.py
- src/api/service.py
- src/api/server.py

## 12. Recommended Next Engineering Steps

1. Run full default experiment to completion in background and archive summary.
2. Add second dataset adapter (BGL/OpenStack) and enable cross-system config.
3. Add time-sliced sequence generation for richer trajectory dynamics.
4. Improve counterfactual optimization with constrained latent-space optimization.
5. Add automated tests in tests/ for preprocessors, vectorizer, models, and API contracts.
6. Add run manifest (git hash, timestamp, environment snapshot) for stricter experiment traceability.

## 13. Commands (Local Venv Workflow)

Setup:

- python3 -m venv .venv
- source .venv/bin/activate
- python -m pip install -e .

Smoke run:

- .venv/bin/python experiments/run_pipeline.py --config configs/smoke.yaml

Temporal smoke run:

- .venv/bin/python experiments/run_pipeline.py --config configs/temporal_smoke.yaml

Temporal full-scale run:

- .venv/bin/python experiments/run_pipeline.py --config configs/temporal_full_scale.yaml

Temporal hybrid run:

- .venv/bin/python experiments/run_pipeline.py --config configs/temporal_hybrid.yaml

Default run:

- .venv/bin/python experiments/run_pipeline.py --config configs/default.yaml

API:

- .venv/bin/python -m api.server
