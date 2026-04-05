from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from analysis import (
    build_event_causal_graph,
    build_trajectories,
    cluster_failures,
    counterfactual_event_shift,
    graph_to_dict,
    latent_velocity,
    reduce_latent,
    summarize_failure_clusters,
    trajectory_risk_score,
)
from evaluation import compute_binary_metrics, plot_confusion_matrix, plot_latent_2d, plot_roc, threshold_by_f1_optimization, threshold_by_percentile
from features import BagOfEventsVectorizer, SequenceAwareVectorizer
from models import BaselineRunner, TemporalVAEConfig, TemporalVAETrainer, VAEConfig, VAETrainer
from preprocessing import HDFSPreprocessor, HDFSXuSplitPreprocessor, SyntheticInjector
from utils.config import load_yaml
from utils.io import ensure_dir, write_json
from utils.repro import set_seed


def load_primary_dataset(cfg: dict):
    dcfg = cfg["dataset"]
    root = Path(dcfg["root"])

    if dcfg.get("format") == "pregrouped_sequences":
        pre = HDFSXuSplitPreprocessor(
            data_root=root,
            train_file=dcfg["train_sequences"],
            normal_file=dcfg["test_normal_sequences"],
            abnormal_file=dcfg["test_abnormal_sequences"],
        )
    else:
        pre = HDFSPreprocessor(
            data_root=root,
            raw_log=dcfg["raw_log"],
            labels_csv=dcfg["labels"],
            structured_log=dcfg.get("structured_log"),
        )

    if isinstance(pre, HDFSXuSplitPreprocessor):
        return pre.build_split_dataset(), {}

    ds = pre.build_sequences()
    return ds, ds.templates


def maybe_load_second_dataset(cfg: dict) -> tuple[list[list[str]], np.ndarray] | None:
    second = cfg["dataset"].get("second_dataset", {})
    if not second.get("enabled", False):
        return None

    root = Path(second["root"])
    raw = root / second.get("raw_log", "")
    labels = root / second.get("labels", "")
    if not raw.exists() or not labels.exists():
        return None

    pre = HDFSPreprocessor(
        data_root=root,
        raw_log=second["raw_log"],
        labels_csv=second["labels"],
        structured_log=second.get("structured_log"),
    )
    ds = pre.build_sequences()
    return ds.sequences, np.array(ds.labels, dtype=int)


def evaluate_model(name: str, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    m = compute_binary_metrics(y, scores, threshold)
    return {
        "model": name,
        "threshold": threshold,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "roc_auc": m.roc_auc,
        "tn": m.tn,
        "fp": m.fp,
        "fn": m.fn,
        "tp": m.tp,
    }


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))

    out_root = ensure_dir(Path(cfg["outputs"]["root"]) / cfg["outputs"]["run_name"])
    plots_dir = ensure_dir(out_root / "plots")
    artifacts_dir = ensure_dir(out_root / "artifacts")

    print("[1/10] Loading dataset...")
    primary_ds, templates = load_primary_dataset(cfg)
    if hasattr(primary_ds, "train_sequences"):
        train_block_ids = primary_ds.train_block_ids
        train_sequences = primary_ds.train_sequences
        train_labels = np.array(primary_ds.train_labels, dtype=int)
        eval_block_ids = primary_ds.eval_block_ids
        eval_sequences = primary_ds.eval_sequences
        eval_labels = np.array(primary_ds.eval_labels, dtype=int)
    else:
        train_block_ids = primary_ds.block_ids
        train_sequences = primary_ds.sequences
        train_labels = np.array(primary_ds.labels, dtype=int)
        eval_block_ids = primary_ds.block_ids
        eval_sequences = primary_ds.sequences
        eval_labels = np.array(primary_ds.labels, dtype=int)

    print(f"Loaded {len(train_sequences)} train sequences and {len(eval_sequences)} eval sequences")
    print(f"Training anomalies: {int(np.sum(train_labels))}, evaluation anomalies: {int(np.sum(eval_labels))}")

    dataset_cfg = cfg.get("dataset", {})
    train_limit = dataset_cfg.get("limit_train_sequences")
    if train_limit is not None:
        train_limit = int(train_limit)
        train_sequences = train_sequences[:train_limit]
        train_labels = train_labels[:train_limit]
        train_block_ids = train_block_ids[:train_limit]

    eval_normal_limit = dataset_cfg.get("limit_eval_normal_sequences")
    eval_abnormal_limit = dataset_cfg.get("limit_eval_abnormal_sequences")
    if eval_normal_limit is not None or eval_abnormal_limit is not None:
        normal_idx = np.where(eval_labels == 0)[0]
        abnormal_idx = np.where(eval_labels == 1)[0]
        if eval_normal_limit is not None:
            normal_idx = normal_idx[: int(eval_normal_limit)]
        if eval_abnormal_limit is not None:
            abnormal_idx = abnormal_idx[: int(eval_abnormal_limit)]
        keep_idx = np.sort(np.concatenate([normal_idx, abnormal_idx]))
        eval_sequences = [eval_sequences[i] for i in keep_idx]
        eval_labels = eval_labels[keep_idx]
        eval_block_ids = [eval_block_ids[i] for i in keep_idx]

    validation_sequences: list[list[str]] = []
    validation_labels = np.array([], dtype=int)
    if hasattr(primary_ds, "split_train_validation"):
        validation_ratio = float(cfg["vae"].get("validation_split", 0.2))
        train_sequences, train_labels_list, validation_sequences, validation_labels_list = primary_ds.split_train_validation(
            validation_ratio=validation_ratio
        )
        train_labels = np.array(train_labels_list, dtype=int)
        validation_labels = np.array(validation_labels_list, dtype=int)
        print(f"Training sequences after split: {len(train_sequences)}, validation sequences: {len(validation_sequences)}")

    if cfg.get("synthetic_injection", {}).get("enabled", False):
        print("[2/10] Applying synthetic anomaly injection...")
        inj_cfg = cfg["synthetic_injection"]
        inj = SyntheticInjector(seed=cfg["seed"])
        inj_res = inj.inject(
            sequences=train_sequences,
            labels=train_labels.tolist(),
            ratio=float(inj_cfg.get("ratio", 0.1)),
            modes=list(inj_cfg.get("modes", ["insert", "remove", "shuffle"])),
        )
        train_sequences = inj_res.sequences
        train_labels = np.array(inj_res.labels, dtype=int)
        print(f"Injected anomalies into {len(inj_res.injected_indices)} sequences")

    representation = cfg["features"].get("representation", "bag_of_events")
    print(f"[3/10] Building features using representation={representation}...")

    if representation == "temporal":
        temporal_vec = SequenceAwareVectorizer(
            min_count=int(cfg["features"]["min_count"]),
            max_vocab_size=cfg["features"].get("max_vocab_size"),
            max_sequence_length=cfg["features"].get("max_sequence_length"),
            unknown_token=cfg["features"].get("unknown_token", "<UNK>"),
            positional_encoding_type=cfg["features"].get("positional_encoding_type", "absolute"),
        )
        train_batch = temporal_vec.fit_transform(train_sequences)
        val_batch = temporal_vec.transform(validation_sequences) if len(validation_sequences) else temporal_vec.transform(train_sequences[:1])
        eval_batch = temporal_vec.transform(eval_sequences)
        print(f"Temporal train tensor shape: {train_batch.token_ids.shape}")
        print(f"Temporal eval tensor shape: {eval_batch.token_ids.shape}")

        y_train_normal = train_labels == 0
        print("[4/10] Training baseline models on bag-of-events comparison features...")
        baseline_vec = BagOfEventsVectorizer(
            min_count=int(cfg["features"]["min_count"]),
            norm=cfg["features"].get("normalize", "l2"),
            max_vocab_size=cfg["features"].get("max_vocab_size"),
            unknown_token=cfg["features"].get("unknown_token", "<UNK>"),
        )
        X_train = baseline_vec.fit_transform(train_sequences).X
        X_eval = baseline_vec.transform(eval_sequences)
        baseline = BaselineRunner(cfg["baselines"]["isolation_forest"], cfg["baselines"]["pca"])
        baseline.fit_normal(X_train, train_labels)
        b_scores = baseline.score(X_eval)
        train_baseline_scores = baseline.score(X_train)
        if_thr = threshold_by_percentile(train_baseline_scores.isolation_forest[y_train_normal], 95)
        pca_thr = threshold_by_percentile(train_baseline_scores.pca_recon[y_train_normal], 95)
        if_eval = evaluate_model("isolation_forest", eval_labels, b_scores.isolation_forest, if_thr)
        pca_eval = evaluate_model("pca_reconstruction", eval_labels, b_scores.pca_recon, pca_thr)

        best_vae = None
        best_vae_eval = None
        best_vae_cfg = None
        best_scores = None
        best_threshold = None

        print("[5/10] Training temporal VAE hyperparameter grid...")
        total_runs = len(cfg["vae"]["latent_dims"]) * len(cfg["vae"]["beta_values"])
        run_id = 0
        for zdim in cfg["vae"]["latent_dims"]:
            for beta in cfg["vae"]["beta_values"]:
                run_id += 1
                print(f"  Temporal VAE run {run_id}/{total_runs}: latent_dim={zdim}, beta={beta}")
                vae_cfg = TemporalVAEConfig(
                    vocab_size=len(temporal_vec.vocab),
                    pad_index=temporal_vec.pad_index,
                    unknown_index=temporal_vec.unknown_index,
                    embedding_dim=int(cfg["vae"].get("embedding_dim", 64)),
                    hidden_dim=int(cfg["vae"]["hidden_dim"]),
                    latent_dim=int(zdim),
                    beta=float(beta),
                    lr=float(cfg["vae"]["learning_rate"]),
                    epochs=int(cfg["vae"]["epochs"]),
                    batch_size=int(cfg["vae"]["batch_size"]),
                    beta_warmup_epochs=int(cfg["vae"].get("beta_warmup_epochs", 0)),
                )
                trainer = TemporalVAETrainer(vae_cfg)
                _ = trainer.fit(train_batch.token_ids[y_train_normal], train_batch.mask[y_train_normal])
                train_scores = trainer.reconstruction_error(train_batch.token_ids, train_batch.mask)
                val_scores = trainer.reconstruction_error(val_batch.token_ids, val_batch.mask)
                eval_scores = trainer.reconstruction_error(eval_batch.token_ids, eval_batch.mask)
                threshold, validation_f1 = threshold_by_f1_optimization(
                    validation_labels if len(validation_labels) else train_labels,
                    val_scores,
                )
                ev = evaluate_model(f"temporal_vae_z{zdim}_b{beta}", eval_labels, eval_scores, threshold)
                ev["validation_f1"] = float(validation_f1)

                if best_vae_eval is None or float(ev["f1"]) > float(best_vae_eval["f1"]):
                    best_vae = trainer
                    best_vae_eval = ev
                    best_vae_cfg = vae_cfg
                    best_scores = eval_scores
                    best_threshold = threshold
                    print(f"    New best temporal VAE F1={float(ev['f1']):.4f}")

        assert best_vae is not None and best_vae_eval is not None and best_vae_cfg is not None and best_scores is not None and best_threshold is not None

        print("[6/10] Computing latent representations...")
        z = best_vae.latent(eval_batch.token_ids, eval_batch.mask)
        event_names = [event for event, _ in sorted(temporal_vec.vocab.items(), key=lambda item: item[1])]
        feature_space_info = {
            "representation": representation,
            "vocab_size": int(len(temporal_vec.vocab)),
            "unknown_token": temporal_vec.unknown_token,
            "pad_token": temporal_vec.pad_token,
            "max_sequence_length": temporal_vec.max_sequence_length,
            "embedding_dim": int(best_vae_cfg.embedding_dim),
        }
        feature_export = {
            "X_train": X_train,
            "X_eval": X_eval,
            "X_payload": eval_batch.token_ids,
            "train_mask": train_batch.mask,
            "eval_mask": eval_batch.mask,
        }
        active_vectorizer = temporal_vec
    else:
        print("[4/10] Training baseline models...")
        vec = BagOfEventsVectorizer(
            min_count=int(cfg["features"]["min_count"]),
            norm=cfg["features"].get("normalize", "l2"),
            max_vocab_size=cfg["features"].get("max_vocab_size"),
            unknown_token=cfg["features"].get("unknown_token", "<UNK>"),
        )
        X_train = vec.fit_transform(train_sequences).X
        X_eval = vec.transform(eval_sequences)
        print(f"Train feature matrix shape: {X_train.shape}")
        print(f"Eval feature matrix shape: {X_eval.shape}")

        y_train_normal = train_labels == 0
        baseline = BaselineRunner(cfg["baselines"]["isolation_forest"], cfg["baselines"]["pca"])
        baseline.fit_normal(X_train, train_labels)
        b_scores = baseline.score(X_eval)

        train_baseline_scores = baseline.score(X_train)
        if_thr = threshold_by_percentile(train_baseline_scores.isolation_forest[y_train_normal], 95)
        pca_thr = threshold_by_percentile(train_baseline_scores.pca_recon[y_train_normal], 95)

        if_eval = evaluate_model("isolation_forest", eval_labels, b_scores.isolation_forest, if_thr)
        pca_eval = evaluate_model("pca_reconstruction", eval_labels, b_scores.pca_recon, pca_thr)

        best_vae = None
        best_vae_eval = None
        best_vae_cfg = None
        best_scores = None
        best_threshold = None

        print("[5/10] Training VAE hyperparameter grid...")
        total_runs = len(cfg["vae"]["latent_dims"]) * len(cfg["vae"]["beta_values"])
        run_id = 0
        for zdim in cfg["vae"]["latent_dims"]:
            for beta in cfg["vae"]["beta_values"]:
                run_id += 1
                print(f"  VAE run {run_id}/{total_runs}: latent_dim={zdim}, beta={beta}")
                vae_cfg = VAEConfig(
                    input_dim=X_train.shape[1],
                    hidden_dim=int(cfg["vae"]["hidden_dim"]),
                    latent_dim=int(zdim),
                    beta=float(beta),
                    lr=float(cfg["vae"]["learning_rate"]),
                    epochs=int(cfg["vae"]["epochs"]),
                    batch_size=int(cfg["vae"]["batch_size"]),
                )
                trainer = VAETrainer(vae_cfg)
                _ = trainer.fit(X_train[y_train_normal])
                scores = trainer.reconstruction_error(X_eval)
                train_scores = trainer.reconstruction_error(X_train)
                threshold, validation_f1 = threshold_by_f1_optimization(
                    train_labels[y_train_normal],
                    train_scores[y_train_normal],
                )
                ev = evaluate_model(f"vae_z{zdim}_b{beta}", eval_labels, scores, threshold)
                ev["validation_f1"] = float(validation_f1)

                if best_vae_eval is None or float(ev["f1"]) > float(best_vae_eval["f1"]):
                    best_vae = trainer
                    best_vae_eval = ev
                    best_vae_cfg = vae_cfg
                    best_scores = scores
                    best_threshold = threshold
                    print(f"    New best VAE F1={float(ev['f1']):.4f}")

        assert best_vae is not None and best_vae_eval is not None and best_vae_cfg is not None and best_scores is not None and best_threshold is not None

        print("[6/10] Computing latent representations...")
        z = best_vae.latent(X_eval)
        event_names = [event for event, _ in sorted(vec.vocab.items(), key=lambda item: item[1])]
        feature_space_info = {
            "representation": representation,
            "vocab_size": int(len(vec.vocab)),
            "unknown_token": vec.unknown_token,
            "max_vocab_size": vec.max_vocab_size,
        }
        feature_export = {
            "X_train": X_train,
            "X_eval": X_eval,
        }
        active_vectorizer = vec

    embed_method = cfg["analysis"].get("embedding_method", "umap")
    max_embed = int(cfg["analysis"].get("max_points_for_embedding", len(z)))
    if len(z) > max_embed:
        embed_idx = np.sort(rng.choice(len(z), size=max_embed, replace=False))
    else:
        embed_idx = np.arange(len(z))

    print(f"[7/10] Building 2D latent embedding with method={embed_method} over {len(embed_idx)} points...")
    embed = reduce_latent(z[embed_idx], method=embed_method, random_state=cfg["seed"])

    anomaly_mask = eval_labels == 1
    z_anom = z[anomaly_mask]
    seq_anom = [s for s, flag in zip(eval_sequences, anomaly_mask) if flag]
    print("[8/10] Running failure typing and trajectory analysis...")
    clusters = cluster_failures(
        z_anom,
        method=cfg["analysis"].get("failure_clustering", "dbscan"),
        eps=float(cfg["analysis"].get("dbscan_eps", 0.8)),
        min_samples=int(cfg["analysis"].get("dbscan_min_samples", 5)),
        k=int(cfg["analysis"].get("kmeans_k", 4)),
    )
    cluster_summary = summarize_failure_clusters(clusters, seq_anom)

    traj = build_trajectories(z, eval_block_ids)
    all_vel = latent_velocity(z)
    fail_vel = latent_velocity(z[anomaly_mask]) if np.any(anomaly_mask) else np.zeros((0, z.shape[1]))
    risk = trajectory_risk_score(all_vel, fail_vel)

    def score_fn(xbatch: np.ndarray) -> np.ndarray:
        if representation == "temporal":
            raise ValueError("Counterfactual scoring for temporal representation is not implemented yet")
        return best_vae.reconstruction_error(xbatch)

    cf_examples = []
    anom_idx = np.where(anomaly_mask)[0][:10]
    for idx in anom_idx:
        if representation != "temporal":
            cf = counterfactual_event_shift(X_eval[idx], score_fn=score_fn, event_names=event_names)
            cf_examples.append({"index": int(idx), "block_id": eval_block_ids[idx], "counterfactual": cf})

    causal_graph = build_event_causal_graph(eval_sequences, eval_labels.tolist(), top_k=30)

    second_eval = None
    second_data = maybe_load_second_dataset(cfg)
    if second_data is not None:
        print("[9/10] Running cross-system evaluation...")
        seq2, y2 = second_data
        if representation == "temporal":
            second_batch = active_vectorizer.transform(seq2)
            s2 = best_vae.reconstruction_error(second_batch.token_ids, second_batch.mask)
        else:
            X2 = active_vectorizer.transform(seq2)
            s2 = best_vae.reconstruction_error(X2)
        thr2 = float(best_vae_eval["threshold"])
        second_eval = evaluate_model("cross_system_vae", y2, s2, thr2)

    # Artifacts for API usage and reproducibility.
    np.save(artifacts_dir / "X.npy", X_eval)
    np.save(artifacts_dir / "X_train.npy", feature_export["X_train"])
    np.save(artifacts_dir / "X_eval.npy", feature_export["X_eval"])
    np.save(artifacts_dir / "y.npy", eval_labels)
    np.save(artifacts_dir / "y_train.npy", train_labels)
    np.save(artifacts_dir / "latent.npy", z)
    np.save(artifacts_dir / "latent_embed_indices.npy", embed_idx)
    np.save(artifacts_dir / "scores_vae.npy", best_scores)
    write_json(
        {
            "vocab": active_vectorizer.vocab,
            "unknown_index": getattr(active_vectorizer, "unknown_index", None),
            "max_vocab_size": getattr(active_vectorizer, "max_vocab_size", None),
            "train_rows": int(len(train_sequences)),
            "eval_rows": int(len(eval_sequences)),
            "representation": representation,
            "validation_rows": int(len(validation_sequences)),
            "decision_threshold": float(best_threshold),
        },
        artifacts_dir / "vocab.json",
    )

    import torch

    torch.save({"state_dict": best_vae.model.state_dict(), "config": best_vae_cfg.__dict__}, artifacts_dir / "vae.pt")

    cm_if = confusion_matrix(eval_labels, (b_scores.isolation_forest >= if_thr).astype(int), labels=[0, 1])
    cm_pca = confusion_matrix(eval_labels, (b_scores.pca_recon >= pca_thr).astype(int), labels=[0, 1])
    cm_vae = confusion_matrix(eval_labels, (best_scores >= float(best_vae_eval["threshold"])).astype(int), labels=[0, 1])

    plot_roc(eval_labels, b_scores.isolation_forest, plots_dir / "roc_isolation_forest.png", "Isolation Forest ROC")
    plot_roc(eval_labels, b_scores.pca_recon, plots_dir / "roc_pca.png", "PCA ROC")
    plot_roc(eval_labels, best_scores, plots_dir / "roc_vae.png", "VAE ROC")
    plot_confusion_matrix(cm_if, plots_dir / "cm_isolation_forest.png", "Isolation Forest")
    plot_confusion_matrix(cm_pca, plots_dir / "cm_pca.png", "PCA Reconstruction")
    plot_confusion_matrix(cm_vae, plots_dir / "cm_vae.png", "VAE")
    plot_latent_2d(embed, eval_labels[embed_idx], plots_dir / "latent_space.png", "Latent Space Projection")

    summary = {
        "dataset": cfg["dataset"]["name"],
        "num_sequences_train": int(len(train_sequences)),
        "num_sequences_eval": int(len(eval_sequences)),
        "num_anomalies_train": int(np.sum(train_labels)),
        "num_anomalies_eval": int(np.sum(eval_labels)),
        "results": {
            "isolation_forest": if_eval,
            "pca": pca_eval,
            "vae_best": best_vae_eval,
            "cross_system": second_eval,
        },
        "vae_best_config": best_vae_cfg.__dict__,
        "trajectory": {
            "num_trajectories": len(traj),
            "global_risk_score": risk,
        },
        "embedding": {
            "method": embed_method,
            "num_points": int(len(embed_idx)),
            "sampled": bool(len(embed_idx) < len(z)),
        },
        "feature_space": feature_space_info,
        "failure_typing": cluster_summary,
        "counterfactual_examples": cf_examples,
        "causal_graph": graph_to_dict(causal_graph),
        "templates": templates,
    }

    print("[10/10] Writing summary and finishing...")
    write_json(summary, out_root / "summary.json")
    print(f"Run complete. Summary at: {out_root / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeuroSys experiment pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
