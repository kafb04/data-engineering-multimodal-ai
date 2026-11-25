"""Trechos de repouso e de intenção de cada tentativa e divisão deixando uma pessoa de fora."""
from __future__ import annotations

import random
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess

from src.ingest import CLASS_MAP, DATASET_NAME, build_fact_table

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
N_TRIALS = 5184
TRIAL_S = 4.0


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def folds():
    """(teste, validação, treino) por pessoa: a seguinte na lista valida e as outras 7 treinam."""
    splits = []
    for i, test in enumerate(SUBJECTS):
        val = SUBJECTS[(i + 1) % len(SUBJECTS)]
        splits.append((test, val, sorted(set(SUBJECTS) - {test, val})))
    return splits


def _scale(data, factor):
    return data * factor


def load_samples(sfreq, band, scale, rest, intention):
    """Trechos de repouso e de intenção, em segundos relativos à seta, das 5.184 tentativas.

    Devolve X (repousos seguidos das intenções), y (0 = repouso, 1 = intenção), os metadados de cada
    linha de X e os nomes dos canais.
    """
    mne.set_log_level("WARNING")
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(range(1, 10)))
    preprocess(dataset, [
        Preprocessor("pick", picks="eeg"),
        Preprocessor(_scale, factor=scale),
        Preprocessor("filter", l_freq=band[0], h_freq=band[1]),
        Preprocessor("resample", sfreq=sfreq),
    ])
    start = rest[0]
    # o fim da janela é contado a partir do fim da tentativa na fonte (seta + 4 s)
    windows = create_windows_from_events(
        dataset, trial_start_offset_samples=round(start * sfreq),
        trial_stop_offset_samples=round((intention[1] - TRIAL_S) * sfreq),
        preload=True, mapping=CLASS_MAP,
    )
    trials = build_fact_table(windows)[["trial_id", "subject_id"]]
    fact = pd.read_parquet(ROOT / "data" / "processed" / "fact_trial.parquet",
                           columns=["trial_id", "has_artifact"])
    trials = trials.merge(fact, on="trial_id", how="left", validate="one_to_one")
    assert len(trials) == N_TRIALS and trials["has_artifact"].notna().all()

    X = np.stack([windows[i][0] for i in range(len(windows))]).astype(np.float32)
    assert X.shape[2] == round((intention[1] - start) * sfreq)

    def cut(begin, end):
        return X[:, :, round((begin - start) * sfreq):round((end - start) * sfreq)]

    meta = pd.concat([trials.assign(window="repouso", y_true=0),
                      trials.assign(window="intencao", y_true=1)], ignore_index=True)
    meta.insert(0, "sample_id", meta["trial_id"] + "_" + meta["window"])
    X = np.concatenate([cut(*rest), cut(*intention)])
    return X, meta["y_true"].to_numpy(np.int64), meta, windows.datasets[0].raw.ch_names
