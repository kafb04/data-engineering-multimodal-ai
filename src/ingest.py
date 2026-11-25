"""Ingestão do BCI IV 2a em tabelas do esquema estrela (Parquet), uma pessoa por vez.

Uso: `python -m src.ingest`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from moabb.datasets import BNCI2014_001

DATASET_NAME = "BNCI2014_001"
TRIALS_PER_SUBJECT = 576
FACT_SCHEMA = pa.schema([
    ("trial_id", pa.string()), ("subject_id", pa.string()),
    ("session_id", pa.string()), ("run_id", pa.string()),
    ("class_id", pa.int16()), ("trial_in_run", pa.int16()),
    ("onset_s", pa.float64()), ("duration_s", pa.float64()),
    ("n_samples", pa.int32()), ("has_artifact", pa.bool_()),
])
SEX = {1: "M", 2: "F"}
CLASS_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
BODY_PART = {"left_hand": "hand_left", "right_hand": "hand_right",
             "feet": "feet", "tongue": "tongue"}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_windows(subject_id):
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[subject_id])
    return create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=False, mapping=CLASS_MAP,
    )


def _artifact_onsets(subject_id):
    """Amostra da seta das tentativas marcadas com ruído, por (sessão, run).

    O braindecode descarta essa marcação, então ela é lida direto do MOABB.
    """
    sessions = BNCI2014_001(artifact_handling="annotate").get_data([subject_id])[subject_id]
    runs = [(session, run, raw) for session, by_run in sessions.items() for run, raw in by_run.items()]
    return {
        (session, run): np.round(
            raw.annotations.onset[raw.annotations.description == "bnci_artifact"] * raw.info["sfreq"]
        ).astype(int)
        for session, run, raw in runs
    }


def _subject_row(recording):
    subject_info = recording.raw.info["subject_info"]
    return {
        "subject_id": f"A{int(recording.description['subject']):02d}",
        "age": recording.raw.info["meas_date"].year - subject_info["birthday"].year,
        "sex": SEX[subject_info["sex"]],
    }


def build_fact_table(windows, artifact_onsets=None) -> pd.DataFrame:
    """Fato: uma linha por tentativa, na mesma ordem das janelas."""
    artifact_onsets = artifact_onsets or {}
    frames = []
    for recording in windows.datasets:
        info, trials = recording.description, recording.metadata
        subject, session, run = f"A{int(info['subject']):02d}", str(info["session"]), str(info["run"])
        start = trials["i_start_in_trial"].to_numpy()
        n_samples = trials["i_stop_in_trial"].to_numpy() - start
        trial_in_run = trials["i_trial_in_dataset"].to_numpy()
        sfreq = recording.raw.info["sfreq"]
        frames.append(pd.DataFrame({
            "trial_id": f"{subject}_{session}_run_{run}_t"
                        + pd.Series(trial_in_run).astype(str).str.zfill(2),
            "subject_id": subject,
            "session_id": session,
            "run_id": f"run_{run}",
            "class_id": trials["target"].to_numpy(),
            "trial_in_run": trial_in_run,
            "onset_s": start / sfreq,
            "duration_s": n_samples / sfreq,
            "n_samples": n_samples,
            "has_artifact": np.isin(start, artifact_onsets.get((session, run), [])),
        }))
    return pd.concat(frames, ignore_index=True)


def build_dimension_tables(fact_table: pd.DataFrame, subjects):
    dim_subject = pd.DataFrame(subjects).astype(
        {"subject_id": "string", "age": "Int16", "sex": "string"})
    dim_subject["handedness"] = pd.array([pd.NA] * len(dim_subject), dtype="string")

    dim_class = pd.DataFrame({
        "class_id": pd.array(list(CLASS_MAP.values()), dtype="int16"),
        "class_name": pd.array(list(CLASS_MAP), dtype="string"),
        "body_part": pd.array([BODY_PART[name] for name in CLASS_MAP], dtype="string"),
        "paradigm": pd.array(["motor_imagery"] * len(CLASS_MAP), dtype="string"),
    })

    sessions = sorted(fact_table["session_id"].unique())
    dim_session = pd.DataFrame({
        "session_id": pd.array(sessions, dtype="string"),
        "session_role": pd.array(
            ["train" if "train" in session else "test" for session in sessions], dtype="string"),
    })

    runs = sorted(fact_table["run_id"].unique())
    dim_run = pd.DataFrame({
        "run_id": pd.array(runs, dtype="string"),
        "run_number": pd.array([int(run.split("_")[1]) for run in runs], dtype="int16"),
    })

    return {"dim_subject": dim_subject, "dim_class": dim_class,
            "dim_session": dim_session, "dim_run": dim_run}


def ingest():
    """Grava a fato em lotes (um row group por pessoa) e depois as dimensões."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fact_path = OUTPUT_DIR / "fact_trial.parquet"
    subjects = []
    with pq.ParquetWriter(fact_path, FACT_SCHEMA) as writer:
        for subject_id in range(1, 10):
            windows = _load_windows(subject_id)
            batch = build_fact_table(windows, _artifact_onsets(subject_id))
            subjects.append(_subject_row(windows.datasets[0]))
            writer.write_table(pa.Table.from_pandas(batch, schema=FACT_SCHEMA, preserve_index=False))

    fact_table = pd.read_parquet(fact_path)
    assert len(fact_table) == 9 * TRIALS_PER_SUBJECT
    assert fact_table["trial_id"].is_unique

    dimensions = build_dimension_tables(fact_table, subjects)
    for name, table in dimensions.items():
        table.to_parquet(OUTPUT_DIR / f"{name}.parquet", index=False)
    fact_table.to_csv(OUTPUT_DIR / "fact_trial.csv", index=False)
    return {"fact_trial": fact_table, **dimensions}


def main():
    for name, table in ingest().items():
        print(f"{name}: {len(table)} linhas, {len(table.columns)} colunas "
              f"-> data/processed/{name}.parquet")


if __name__ == "__main__":
    main()
