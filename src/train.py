"""Treino do EEGNet no BCI IV 2a com divisão por sujeito (GroupKFold, 9 folds) e predições.

Uso: `python -m src.train`.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGNet
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from src.ingest import CLASS_MAP, DATASET_NAME, build_fact_table

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PREDICTIONS_PATH = ROOT / "data" / "predictions" / "predictions_eegnet.parquet"
MODEL_VERSION = "eegnet_braindecode-1.8.1_groupkfold9_seed42_v2"
SEED, N_FOLDS, N_EPOCHS, BATCH_SIZE, LR = 42, 9, 80, 64, 1e-3
SFREQ = 128  # Hz, como no artigo do EEGNet
CLASS_NAMES = sorted(CLASS_MAP, key=CLASS_MAP.get)


def load_windows():
    """Épocas pré-processadas dos 9 sujeitos + metadados alinhados por posição."""
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(range(1, 10)))
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, eog=False, stim=False),
        Preprocessor(lambda data: data * 1e6),  # V -> µV
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        Preprocessor("resample", sfreq=SFREQ),
    ])
    windows = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=True, mapping=CLASS_MAP,
    )
    # trial_id na mesma ordem de windows[i]
    meta = build_fact_table(windows, set())[["trial_id", "subject_id", "session_id", "class_id"]]
    fact = pd.read_parquet(ROOT / "data" / "processed" / "fact_trial.parquet",
                           columns=["trial_id", "has_artifact"])
    meta = meta.merge(fact, on="trial_id", how="left", validate="one_to_one")
    assert len(meta) == 5184 and meta["has_artifact"].notna().all()

    X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    y = np.array([windows[i][1] for i in range(len(windows))], dtype=np.int64)
    assert (y == meta["class_id"].to_numpy()).all()
    return X, y, meta


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(torch.from_numpy(X[i:i + 256])), dim=1)
                          for i in range(0, len(X), 256)]).numpy()


def train_fold(X, y, train_idx, val_idx):
    torch.manual_seed(SEED)  # mesma inicialização em todos os folds
    model = EEGNet(n_chans=X.shape[1], n_outputs=len(CLASS_NAMES), n_times=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    generator = torch.Generator().manual_seed(SEED)
    best_score, best_state = -1.0, None
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in torch.randperm(len(train_idx), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss_fn(model(X_train[batch]), y_train[batch]).backward()
            optimizer.step()
        # guarda a melhor época na validação
        score = balanced_accuracy_score(y[val_idx], predict(model, X[val_idx]).argmax(1))
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        print(f"  época {epoch + 1:3d}: acc. balanceada val = {score:.3f}", flush=True)
    model.load_state_dict(best_state)
    return model, best_score


def main():
    torch.set_num_threads(8)
    X, y, meta = load_windows()
    clean = ~meta["has_artifact"].to_numpy()
    subjects = meta["subject_id"].to_numpy()
    MODELS_DIR.mkdir(exist_ok=True)
    predictions = []
    for fold, (train_pool, test_idx) in enumerate(
            GroupKFold(n_splits=N_FOLDS).split(X, y, groups=subjects)):
        pool_subjects = sorted(set(subjects[train_pool]))
        val_subject = pool_subjects[-1]
        train_idx = train_pool[(subjects[train_pool] != val_subject) & clean[train_pool]]
        val_idx = train_pool[(subjects[train_pool] == val_subject) & clean[train_pool]]
        train_s, val_s, test_s = (set(subjects[i]) for i in (train_idx, val_idx, test_idx))
        assert not (train_s & val_s) and not (train_s & test_s) and not (val_s & test_s)
        print(f"fold {fold}: treino {sorted(train_s)} | val {val_subject} | teste {sorted(test_s)}")

        # z-score por canal ajustado no treino
        mean = X[train_idx].mean(axis=(0, 2), keepdims=True)
        std = X[train_idx].std(axis=(0, 2), keepdims=True)
        X_fold = (X - mean) / std

        model, val_score = train_fold(X_fold, y, train_idx, val_idx)
        torch.save(model.state_dict(), MODELS_DIR / f"eegnet_fold{fold}.pt")
        proba = predict(model, X_fold[test_idx])
        fold_pred = meta.iloc[test_idx].reset_index(drop=True)
        fold_pred = fold_pred.rename(columns={"class_id": "y_true"})
        fold_pred.insert(3, "fold", fold)
        fold_pred.insert(4, "model_version", MODEL_VERSION)
        fold_pred["y_pred"] = proba.argmax(1)
        for k, name in enumerate(CLASS_NAMES):
            fold_pred[f"p_{name}"] = proba[:, k].astype(np.float32)
        predictions.append(fold_pred)
        print(f"fold {fold}: melhor val = {val_score:.3f}")

    out = pd.concat(predictions, ignore_index=True).astype(
        {"y_true": "int16", "y_pred": "int16", "fold": "int16"})
    assert len(out) == 5184 and out["trial_id"].is_unique
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"{len(out)} predições -> {PREDICTIONS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
