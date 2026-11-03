"""Detecção de intenção de movimento (imagética × repouso) no BCI IV 2a: EEGConformer e MLP.

Cada trial gera duas amostras: repouso (1,5 s antes do cue) e intenção (1,5 a 3,0 s após o
cue). Divisão fixa por sujeito: treino A01–A05, validação A06–A07 (época e limiar) e teste
A08–A09, avaliado uma única vez.

- EEGConformer (CNN + Transformer) recebe o sinal (22 canais × 192 amostras).
- MLP recebe 44 características: log da potência nas bandas mu e beta de cada canal.

Uso: `python -m src.train`.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess
from scipy.signal import welch
from sklearn.metrics import roc_auc_score, roc_curve

from src.ingest import CLASS_MAP, DATASET_NAME, build_fact_table

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "data" / "predictions"
SEED, BATCH_SIZE = 42, 64
TRAIN_SUBJECTS = ["A01", "A02", "A03", "A04", "A05"]
VAL_SUBJECTS = ["A06", "A07"]
TEST_SUBJECTS = ["A08", "A09"]
SFREQ = 128  # Hz
WINDOW = int(1.5 * SFREQ)  # 192 amostras = 1,5 s
N_TRIALS = 5184
BANDS = {"mu": (8, 12), "beta": (13, 30)}  # Hz

# weight_decay = regularização L2; dropout de 0,5 nos dois modelos
CONFIG = {
    "eegconformer": {"epochs": 40, "lr": 2e-4, "weight_decay": 1e-4, "betas": (0.5, 0.999)},
    "mlp": {"epochs": 100, "lr": 1e-3, "weight_decay": 1e-3, "betas": (0.9, 0.999)},
}


def model_version(name):
    return f"{name}_braindecode-1.8.1_intencao-repouso_split-fixo_seed{SEED}_v3"


def load_samples():
    """Janelas de repouso e de intenção de cada trial + metadados alinhados por posição."""
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(range(1, 10)))
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, eog=False, stim=False),
        Preprocessor(lambda data: data * 1e6),  # V -> µV
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        Preprocessor("resample", sfreq=SFREQ),
    ])
    # de 1,5 s antes do cue até 3,0 s depois (o trial termina 4,0 s após o cue)
    windows = create_windows_from_events(
        dataset, trial_start_offset_samples=-WINDOW, trial_stop_offset_samples=-SFREQ,
        preload=True, mapping=CLASS_MAP,
    )
    trials = build_fact_table(windows, set())[["trial_id", "subject_id"]]  # mesma ordem de windows[i]
    fact = pd.read_parquet(ROOT / "data" / "processed" / "fact_trial.parquet",
                           columns=["trial_id", "has_artifact"])
    trials = trials.merge(fact, on="trial_id", how="left", validate="one_to_one")
    assert len(trials) == N_TRIALS and trials["has_artifact"].notna().all()

    X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    assert X.shape[2] == 3 * WINDOW  # 4,5 s
    rest = X[:, :, :WINDOW]           # cue - 1,5 s até o cue
    intention = X[:, :, 2 * WINDOW:]  # cue + 1,5 s até cue + 3,0 s (a seta some em 1,25 s)

    meta = pd.concat([trials.assign(window="repouso", y_true=0),
                      trials.assign(window="intencao", y_true=1)], ignore_index=True)
    meta.insert(0, "sample_id", meta["trial_id"] + "_" + meta["window"])
    return np.concatenate([rest, intention]), meta["y_true"].to_numpy(np.int64), meta


def band_power(X):
    """Log da potência média por banda e canal (Welch): (amostras, canais × bandas)."""
    freqs, psd = welch(X, fs=SFREQ, nperseg=SFREQ, axis=-1)
    feats = [psd[..., (freqs >= lo) & (freqs <= hi)].mean(-1) for lo, hi in BANDS.values()]
    return np.log(np.concatenate(feats, axis=1)).astype(np.float32)


def make_model(name, n_inputs):
    if name == "eegconformer":
        return EEGConformer(n_chans=n_inputs[0], n_outputs=2, n_times=n_inputs[1])
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs[0], 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2),
    )


def predict(model, X):
    """Saída do modelo para a classe intenção."""
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(torch.from_numpy(X[i:i + 256])), dim=1)[:, 1]
                          for i in range(0, len(X), 256)]).numpy()


def train_model(name, X, y, train_idx, val_idx):
    cfg = CONFIG[name]
    torch.manual_seed(SEED)  # mesma inicialização a cada execução
    model = make_model(name, X.shape[1:])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["betas"],
                                 weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    y_val = torch.from_numpy(y[val_idx])
    generator = torch.Generator().manual_seed(SEED)
    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        losses = []
        for batch in torch.randperm(len(train_idx), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        p_val = predict(model, X[val_idx])
        val_loss = torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(p_val).clamp(1e-6, 1 - 1e-6), y_val.float()).item()
        auc = roc_auc_score(y[val_idx], p_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_auc": auc})
        # parada antecipada: guarda a época com maior AUC na validação
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
        if name == "eegconformer":
            print(f"  época {epoch:3d}: AUC val = {auc:.3f} ({time.time() - start:.0f} s)", flush=True)
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def choose_threshold(y_true, p):
    """Limiar que maximiza a acurácia balanceada (sensibilidade + especificidade)."""
    fpr, tpr, thresholds = roc_curve(y_true, p)
    return float(min(thresholds[np.argmax(tpr - fpr)], 1.0))


def main():
    torch.set_num_threads(8)
    signal, y, meta = load_samples()
    inputs = {"eegconformer": signal, "mlp": band_power(signal)}
    clean = ~meta["has_artifact"].to_numpy()
    subjects = meta["subject_id"].to_numpy()
    assert not (set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS)) and not (set(TEST_SUBJECTS) & (
        set(TRAIN_SUBJECTS) | set(VAL_SUBJECTS)))
    # treino e validação só com trials sem artefato; o teste fica com todos
    train_idx = np.flatnonzero(np.isin(subjects, TRAIN_SUBJECTS) & clean)
    val_idx = np.flatnonzero(np.isin(subjects, VAL_SUBJECTS) & clean)
    test_idx = np.flatnonzero(np.isin(subjects, TEST_SUBJECTS))
    # cada trial entra com as duas janelas (a quarentena tira o trial inteiro): 50% de cada classe
    for idx in (train_idx, val_idx, test_idx):
        assert 2 * y[idx].sum() == len(idx)
    print(f"amostras: treino {len(train_idx)} | validação {len(val_idx)} | teste {len(test_idx)}",
          flush=True)
    MODELS_DIR.mkdir(exist_ok=True)
    predictions, histories = [], []
    for name, X in inputs.items():
        # z-score ajustado só no treino (por canal no sinal, por característica no MLP)
        axes = (0, 2) if X.ndim == 3 else 0
        mean = X[train_idx].mean(axis=axes, keepdims=True)
        std = X[train_idx].std(axis=axes, keepdims=True)
        X_norm = (X - mean) / std

        model, history = train_model(name, X_norm, y, train_idx, val_idx)
        # limiar fixado na validação, antes de olhar o teste
        threshold = choose_threshold(y[val_idx], predict(model, X_norm[val_idx]))
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        histories.append(history.assign(model=name))

        for split, idx in [("validacao", val_idx), ("teste", test_idx)]:
            p = predict(model, X_norm[idx])
            rows = meta.iloc[idx][["sample_id", "trial_id", "subject_id", "window", "y_true"]]
            predictions.append(rows.assign(
                split=split, model=name, model_version=model_version(name),
                p_intencao=p.astype(np.float32), threshold=threshold,
                y_pred=(p >= threshold).astype(np.int16)))
        best = history.loc[history["val_auc"].idxmax()]
        print(f"{name}: melhor época {int(best['epoch'])}, AUC val = {best['val_auc']:.3f}, "
              f"limiar = {threshold:.3f}", flush=True)

    out = pd.concat(predictions, ignore_index=True).astype(
        {"y_true": "int16", "threshold": "float32"})
    out.insert(0, "prediction_id", out["model"] + "_" + out["sample_id"])
    assert len(out) == 2 * (len(val_idx) + len(test_idx)) and out["prediction_id"].is_unique
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "fact_prediction.parquet", index=False)
    pd.concat(histories, ignore_index=True).to_parquet(OUT_DIR / "training_history.parquet", index=False)
    print(f"{len(out)} predições -> data/predictions/fact_prediction.parquet", flush=True)


if __name__ == "__main__":
    main()
