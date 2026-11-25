"""Detecção de intenção de movimento (repouso × intenção) com CSP + LDA e EEGNet.

Cada tentativa gera dois trechos de 1,5 s: repouso (antes da seta) e intenção (de 1,5 s a 3,0 s
depois da seta, quando ela já sumiu). A avaliação deixa uma pessoa de fora: para cada uma das 9,
a seguinte na lista valida (época do EEGNet) e as outras 7 treinam. Treino e validação usam só
tentativas sem ruído; o teste usa todas.

Uso: `python -m src.train`.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGNet
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from src.samples import ROOT, SEED, folds, load_samples, set_seed

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "data" / "predictions"
SFREQ = 128
BAND = (4.0, 38.0)
REST, INTENTION = (-1.5, 0.0), (1.5, 3.0)
N_CSP = 6
EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY = 40, 64, 1e-3, 1e-3


def model_version(name):
    return f"{name}_braindecode-1.8.1_deixa-uma-pessoa-de-fora_seed{SEED}_v4"


def predict(model, X):
    """Saída do EEGNet para a classe intenção."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1)[:, 1].numpy()


def fit_csp_lda(X, y, train_idx):
    return make_pipeline(CSP(n_components=N_CSP, log=True), LinearDiscriminantAnalysis()).fit(
        X[train_idx].astype(np.float64), y[train_idx])


def train_eegnet(X, y, train_idx, val_idx):
    """Treina o EEGNet e devolve os pesos da época com maior área ROC na validação."""
    set_seed()
    model = EEGNet(n_chans=X.shape[1], n_outputs=2, n_times=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    generator = torch.Generator().manual_seed(SEED)
    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in torch.randperm(len(train_idx), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        p_val = predict(model, X[val_idx])
        val_loss = torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(p_val).clamp(1e-6, 1 - 1e-6), torch.from_numpy(y[val_idx]).float()).item()
        auc = roc_auc_score(y[val_idx], p_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_auc": auc})
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def run_fold(X, y, meta, test, val, train):
    subjects, clean = meta["subject_id"].to_numpy(), ~meta["has_artifact"].to_numpy()
    train_idx = np.flatnonzero(np.isin(subjects, train) & clean)
    val_idx = np.flatnonzero((subjects == val) & clean)
    test_idx = np.flatnonzero(subjects == test)

    mean = X[train_idx].mean(axis=(0, 2), keepdims=True)
    std = X[train_idx].std(axis=(0, 2), keepdims=True)
    X_norm = (X - mean) / std

    csp_lda = fit_csp_lda(X, y, train_idx)
    eegnet, history = train_eegnet(X_norm, y, train_idx, val_idx)
    torch.save(eegnet.state_dict(), MODELS_DIR / f"eegnet_{test}.pt")

    rows = meta.iloc[test_idx][["sample_id", "trial_id", "subject_id", "window", "y_true"]]
    scores = {"csp_lda": csp_lda.predict_proba(X[test_idx].astype(np.float64))[:, 1],
              "eegnet": predict(eegnet, X_norm[test_idx])}
    predictions = pd.concat([
        rows.assign(model=name, model_version=model_version(name), val_subject=val,
                    p_intencao=p.astype(np.float32), y_pred=(p >= 0.5).astype(np.int16))
        for name, p in scores.items()])
    aucs = {name: roc_auc_score(rows["y_true"], p) for name, p in scores.items()}
    print(f"{test}: treino {len(train_idx)} | validação {val} | teste {len(test_idx)} amostras | "
          f"AUC CSP + LDA {aucs['csp_lda']:.3f}, EEGNet {aucs['eegnet']:.3f} "
          f"(época {history.loc[history['val_auc'].idxmax(), 'epoch']})", flush=True)
    return predictions, history.assign(test_subject=test, val_subject=val)


def main():
    X, y, meta, _ = load_samples(SFREQ, BAND, 1e6, REST, INTENTION)
    MODELS_DIR.mkdir(exist_ok=True)
    results = [run_fold(X, y, meta, *fold) for fold in folds()]

    out = pd.concat([predictions for predictions, _ in results], ignore_index=True).astype(
        {"y_true": "int16"})
    out.insert(0, "prediction_id", out["model"] + "_" + out["sample_id"])
    assert len(out) == 2 * len(meta) and out["prediction_id"].is_unique
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "fact_prediction.parquet", index=False)
    pd.concat([history for _, history in results], ignore_index=True).to_parquet(
        OUT_DIR / "training_history.parquet", index=False)
    print(f"{len(out)} predições -> data/predictions/fact_prediction.parquet", flush=True)


if __name__ == "__main__":
    main()
