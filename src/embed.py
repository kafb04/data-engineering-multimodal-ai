"""Embeddings do modelo de fundação CBraMod (pré-treinado, congelado) e busca de vizinhos.

Cada tentativa gera dois trechos de 1 s a 200 Hz, o formato do pré-treino: repouso (1 s antes da
seta) e intenção (de 1,5 s a 2,5 s depois). O embedding junta a saída do encoder dos 22 canais
(22 × 200 = 4.400 números), porque a média entre canais apagaria a diferença entre C3 e C4. Duas
buscas de 10 vizinhos por cosseno:

- `todos`: cada trecho contra todos os outros, para achar cópias e medir o peso da pessoa;
- `treino`: cada pessoa contra as tentativas sem ruído das 7 que treinam no mesmo fold de
  `src.train`, para a votação.

Os vetores ficam em `embedding_vectors.parquet`, só local (licença CC BY-ND e ~180 MB).

Uso: `python -m src.embed`.
"""
from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from braindecode.models import CBraMod
from scipy.signal import welch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from src.samples import N_TRIALS, ROOT, SEED, folds, load_samples

OUT_DIR = ROOT / "data" / "embeddings"
HF_REPO = "braindecode/cbramod-pretrained"
HF_REVISION = "584cdc415913739a05d84bf0c1cb3db397764507"
MODEL_VERSION = f"cbramod-pretrained@{HF_REVISION[:7]}_braindecode-1.8.1_encoder-concat_1s-200Hz"
SFREQ = 200
BAND = (0.3, 75.0)
REST, INTENTION = (-1.0, 0.0), (1.5, 2.5)
K = 10
PAIR_CHANNELS = ["C3", "Cz", "C4"]


def embed(X):
    model = CBraMod.from_pretrained(HF_REPO, revision=HF_REVISION, return_encoder_output=True)
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(X[i:i + 256])).flatten(1)
                          for i in range(0, len(X), 256)]).numpy()


def unit(Z, center):
    Z = Z - center
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def neighbors(Z, meta, query_idx, pool_idx, pool_name):
    """K vizinhos de cosseno de cada trecho de query_idx dentro de pool_idx, sem ele mesmo."""
    nn = NearestNeighbors(n_neighbors=K, metric="cosine").fit(Z[pool_idx])
    same_pool = np.array_equal(query_idx, pool_idx)
    dist, pos = nn.kneighbors() if same_pool else nn.kneighbors(Z[query_idx])
    q = meta.iloc[np.repeat(query_idx, K)].reset_index(drop=True)
    n = meta.iloc[pool_idx[pos.ravel()]].reset_index(drop=True)
    return pd.DataFrame({
        "sample_id": q["sample_id"], "subject_id": q["subject_id"], "window": q["window"],
        "pool": pool_name, "rank": np.tile(np.arange(1, K + 1), len(query_idx)),
        "neighbor_sample_id": n["sample_id"], "neighbor_subject_id": n["subject_id"],
        "neighbor_window": n["window"], "cosine_sim": (1 - dist.ravel()).astype(np.float32),
        "same_subject": q["subject_id"] == n["subject_id"],
        "same_trial": q["trial_id"] == n["trial_id"],
        "model_version": MODEL_VERSION,
    })


def plot_pairs(X, meta, channels, pairs):
    """Espectro de C3, Cz e C4 de um trecho e do seu vizinho mais parecido em outra pessoa."""
    position = pd.Series(np.arange(len(meta)), index=meta["sample_id"])
    ids = pairs[["sample_id", "neighbor_sample_id"]].to_numpy()
    picks = [channels.index(c) for c in PAIR_CHANNELS]
    freqs, psd = welch(X[position[ids.ravel()].to_numpy()][:, picks], fs=SFREQ, nperseg=SFREQ)
    band = freqs <= 40
    psd = psd.reshape(len(pairs), 2, len(picks), -1)[..., band]
    fig, axes = plt.subplots(len(pairs), len(picks), figsize=(11, 2.6 * len(pairs)), sharex=True)
    for (row, col), ax in np.ndenumerate(axes):
        ax.semilogy(freqs[band], psd[row, 0, col], "-", label=ids[row, 0])
        ax.semilogy(freqs[band], psd[row, 1, col], "--", label=ids[row, 1])
        similarity = f"{pairs['cosine_sim'].iat[row]:.3f}".replace(".", ",")
        ax.set_title(f"{PAIR_CHANNELS[col]} · cosseno {similarity}", fontsize=9)
    for ax in axes[:, 0]:
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Hz")
    fig.suptitle("Trecho (—) × vizinho mais parecido em outra pessoa (--): "
                 "maior, mediana e menor semelhança")
    fig.tight_layout()
    fig.savefig(ROOT / "figuras" / "pares_vizinhos.png", dpi=120)
    plt.close(fig)


def main():
    X, _, meta, channels = load_samples(SFREQ, BAND, 1e4, REST, INTENTION)
    Z = embed(X)
    assert Z.shape == (2 * N_TRIALS, 22 * SFREQ) and np.isfinite(Z).all()

    subjects, clean = meta["subject_id"].to_numpy(), ~meta["has_artifact"].to_numpy()
    all_idx = np.arange(len(meta))
    Z_all = unit(Z, Z.mean(axis=0))
    tables = [neighbors(Z_all, meta, all_idx, all_idx, "todos")]
    for test, _, train in folds():
        train_idx = np.flatnonzero(np.isin(subjects, train) & clean)
        tables.append(neighbors(unit(Z, Z[train_idx].mean(axis=0)), meta,
                                np.flatnonzero(subjects == test), train_idx, "treino"))
    nb = pd.concat(tables, ignore_index=True)
    assert (nb["sample_id"] != nb["neighbor_sample_id"]).all()
    assert not nb.loc[nb["pool"] == "treino", "same_subject"].any()

    xy = TSNE(n_components=2, init="pca", random_state=SEED).fit_transform(Z_all)
    emb = meta.drop(columns="y_true").assign(
        model_version=MODEL_VERSION, tsne_x=xy[:, 0].astype(np.float32),
        tsne_y=xy[:, 1].astype(np.float32),
        signal_sha1=[hashlib.sha1(x.tobytes()).hexdigest() for x in X])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb.to_parquet(OUT_DIR / "fact_embedding.parquet", index=False)
    nb.to_parquet(OUT_DIR / "fact_neighbor.parquet", index=False)
    pq.write_table(pa.table({
        "sample_id": meta["sample_id"], "subject_id": meta["subject_id"],
        "model_version": [MODEL_VERSION] * len(meta),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(Z_all.ravel()), Z_all.shape[1]),
    }), OUT_DIR / "embedding_vectors.parquet")

    top1 = nb[(nb["pool"] == "treino") & (nb["rank"] == 1)]
    top1 = top1.sort_values("cosine_sim", ignore_index=True)
    plot_pairs(X, meta, channels, top1.iloc[[-1, len(top1) // 2, 0]])
    print(f"{len(emb)} embeddings, {len(nb)} vizinhos -> data/embeddings/", flush=True)


if __name__ == "__main__":
    main()
