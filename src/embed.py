"""Embeddings do BCI IV 2a com o modelo de fundação CBraMod (pré-treinado, congelado) e busca de vizinhos.

Cada trial gera duas amostras de 1 s (o CBraMod usa patches de 1 s a 200 Hz): repouso (1 s antes
do cue) e intenção (1,5 a 2,5 s após o cue). O embedding é a saída do encoder dos 22 canais
concatenados (22 × 200 = 4.400 dimensões; a média sobre canais apagaria a lateralização C3/C4),
centrado pela média do treino e normalizado. Duas buscas de 10 vizinhos por cosseno:

- `todos`: cada amostra contra todas as outras (vizinhos do mesmo sujeito, quase duplicatas);
- `treino`: validação e teste contra só os sujeitos de treino (kNN intenção × repouso).

Os vetores de 4.400 dimensões não são gravados (licença CC BY-ND); só IDs, t-SNE e vizinhos.

Uso: `python -m src.embed`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import MOABBDataset
from braindecode.models import CBraMod
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess
from scipy.signal import welch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from src.ingest import CLASS_MAP, DATASET_NAME, build_fact_table
from src.train import N_TRIALS, SEED, TEST_SUBJECTS, TRAIN_SUBJECTS, VAL_SUBJECTS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "embeddings"
HF_REPO = "braindecode/cbramod-pretrained"
HF_REVISION = "584cdc415913739a05d84bf0c1cb3db397764507"  # commit fixo dos pesos no Hugging Face
MODEL_VERSION = f"cbramod-pretrained@{HF_REVISION[:7]}_braindecode-1.8.1_encoder-concat_1s-200Hz"
SFREQ = 200  # Hz, como no pré-treino
PATCH = SFREQ  # 1 s
K = 10


def load_samples():
    """Janelas de 1 s de repouso e de intenção + metadados (mesmos sample_id de fact_prediction)."""
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(range(1, 10)))
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, eog=False, stim=False),
        Preprocessor(lambda data: data * 1e4),  # V -> µV/100, escala do CBraMod
        Preprocessor("filter", l_freq=0.3, h_freq=75.0),
        Preprocessor("resample", sfreq=SFREQ),
    ])
    # de 1 s antes do cue até 2,5 s depois (o trial termina 4,0 s após o cue)
    windows = create_windows_from_events(
        dataset, trial_start_offset_samples=-PATCH, trial_stop_offset_samples=-int(1.5 * SFREQ),
        preload=True, mapping=CLASS_MAP,
    )
    channels = windows.datasets[0].raw.ch_names
    trials = build_fact_table(windows, set())[["trial_id", "subject_id"]]  # mesma ordem de windows[i]
    fact = pd.read_parquet(ROOT / "data" / "processed" / "fact_trial.parquet",
                           columns=["trial_id", "has_artifact"])
    trials = trials.merge(fact, on="trial_id", how="left", validate="one_to_one")
    assert len(trials) == N_TRIALS and trials["has_artifact"].notna().all()

    X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    assert X.shape[2] == int(3.5 * SFREQ)
    rest = X[:, :, :PATCH]                   # cue - 1 s até o cue
    intention = X[:, :, int(2.5 * PATCH):]   # cue + 1,5 s até cue + 2,5 s (a seta some em 1,25 s)

    meta = pd.concat([trials.assign(window="repouso"), trials.assign(window="intencao")],
                     ignore_index=True)
    meta.insert(0, "sample_id", meta["trial_id"] + "_" + meta["window"])
    split = {**dict.fromkeys(TRAIN_SUBJECTS, "treino"), **dict.fromkeys(VAL_SUBJECTS, "validacao"),
             **dict.fromkeys(TEST_SUBJECTS, "teste")}
    meta["split"] = meta["subject_id"].map(split)
    return np.concatenate([rest, intention]), meta, channels


def embed(X, train_idx):
    model = CBraMod.from_pretrained(HF_REPO, revision=HF_REVISION, return_encoder_output=True)
    model.eval()
    with torch.no_grad():
        # saída do encoder: (amostras, canais, 1 patch, 200) -> canais concatenados
        Z = torch.cat([model(torch.from_numpy(X[i:i + 256])).flatten(1)
                       for i in range(0, len(X), 256)]).numpy()
    Z -= Z[train_idx].mean(axis=0)  # centro estimado só no treino
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def neighbors(Z, meta, query_idx, pool_idx, pool_name):
    """K vizinhos de cosseno de cada amostra de query_idx dentro de pool_idx, sem ela mesma."""
    same_pool = np.array_equal(query_idx, pool_idx)
    nn = NearestNeighbors(n_neighbors=K + same_pool, metric="cosine").fit(Z[pool_idx])
    dist, pos = nn.kneighbors(Z[query_idx])
    if same_pool:
        assert (pos[:, 0] == np.arange(len(query_idx))).all()  # o 1º é a própria amostra
        dist, pos = dist[:, 1:], pos[:, 1:]
    q = meta.iloc[np.repeat(query_idx, K)].reset_index(drop=True)
    n = meta.iloc[pool_idx[pos.ravel()]].reset_index(drop=True)
    return pd.DataFrame({
        "sample_id": q["sample_id"], "subject_id": q["subject_id"], "split": q["split"],
        "window": q["window"], "pool": pool_name, "rank": np.tile(np.arange(1, K + 1), len(query_idx)),
        "neighbor_sample_id": n["sample_id"], "neighbor_subject_id": n["subject_id"],
        "neighbor_split": n["split"], "neighbor_window": n["window"],
        "cosine_sim": (1 - dist.ravel()).astype(np.float32),
        "same_subject": q["subject_id"] == n["subject_id"],
        "same_trial": q["trial_id"] == n["trial_id"],
        "model_version": MODEL_VERSION,
    })


def plot_pairs(X, meta, channels, pairs):
    """PSD de C3/Cz/C4: amostra de teste ao lado do 1º vizinho no treino."""
    picks = [channels.index(c) for c in ("C3", "Cz", "C4")]
    index = pd.Series(np.arange(len(meta)), index=meta["sample_id"])
    fig, axes = plt.subplots(len(pairs), 3, figsize=(11, 2.6 * len(pairs)), sharex=True)
    for row, (_, pair) in zip(axes, pairs.iterrows()):
        for sample_id, style in [(pair["sample_id"], "-"), (pair["neighbor_sample_id"], "--")]:
            freqs, psd = welch(X[index[sample_id], picks], fs=SFREQ, nperseg=PATCH)
            band = freqs <= 40
            for ax, ch, p in zip(row, ("C3", "Cz", "C4"), psd):
                ax.semilogy(freqs[band], p[band], style, label=sample_id)
                ax.set_title(f"{ch} · cosseno {pair['cosine_sim']:.3f}", fontsize=9)
        row[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Hz")
    fig.suptitle("Amostra de teste (—) × 1º vizinho no treino (--): maior, mediana e menor similaridade")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pares_vizinhos.png", dpi=120)


def main():
    torch.set_num_threads(8)
    X, meta, channels = load_samples()
    train_idx = np.flatnonzero(meta["split"] == "treino")
    Z = embed(X, train_idx)
    assert Z.shape == (2 * N_TRIALS, 22 * 200) and np.isfinite(Z).all() and meta["sample_id"].is_unique
    pred = pd.read_parquet(ROOT / "data" / "predictions" / "fact_prediction.parquet", columns=["sample_id"])
    assert set(pred["sample_id"]) <= set(meta.loc[meta["split"] != "treino", "sample_id"])

    all_idx = np.arange(len(meta))
    eval_idx = np.flatnonzero(meta["split"] != "treino")
    nb = pd.concat([neighbors(Z, meta, all_idx, all_idx, "todos"),
                    neighbors(Z, meta, eval_idx, train_idx, "treino")], ignore_index=True)
    assert (nb["sample_id"] != nb["neighbor_sample_id"]).all()
    assert (nb.loc[nb["pool"] == "treino", "neighbor_split"] == "treino").all()

    xy = TSNE(n_components=2, init="pca", random_state=SEED).fit_transform(Z)
    emb = meta.assign(model_version=MODEL_VERSION, tsne_x=xy[:, 0].astype(np.float32),
                      tsne_y=xy[:, 1].astype(np.float32))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb.to_parquet(OUT_DIR / "fact_embedding.parquet", index=False)
    nb.to_parquet(OUT_DIR / "fact_neighbor.parquet", index=False)

    top1 = nb[(nb["pool"] == "treino") & (nb["split"] == "teste") & (nb["rank"] == 1)]
    top1 = top1.sort_values("cosine_sim", ignore_index=True)
    plot_pairs(X, meta, channels, top1.iloc[[-1, len(top1) // 2, 0]])
    print(f"{len(emb)} embeddings, {len(nb)} vizinhos -> data/embeddings/", flush=True)


if __name__ == "__main__":
    main()
