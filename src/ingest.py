"""Ingestão do BCI IV 2a em tabelas do esquema estrela (Parquet).

Uso: `python -m src.ingest`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from moabb.datasets import BNCI2014_001

DATASET_NAME = "BNCI2014_001"
TRIALS_PER_SUBJECT = 576  # 2 sessões × 6 runs × 48 trials
FACT_SCHEMA = pa.schema([
    ("trial_id", pa.string()), ("subject_id", pa.string()),
    ("session_id", pa.string()), ("run_id", pa.string()),
    ("class_id", pa.int16()), ("trial_in_run", pa.int16()),
    ("onset_s", pa.float64()), ("duration_s", pa.float64()),
    ("n_samples", pa.int32()), ("has_artifact", pa.bool_()),
])
SEX = {1: "M", 2: "F"}  # códigos do MNE
CLASS_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
BODY_PART = {"left_hand": "hand_left", "right_hand": "hand_right",
             "feet": "feet", "tongue": "tongue"}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_windows(subject_ids):
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(subject_ids))
    # preload=False: só metadados, sem carregar o sinal.
    return create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=False, mapping=CLASS_MAP,
    )


def _artifact_trials(subject_id):
    """(sessão, run, amostra do cue) dos trials marcados como artefato na fonte."""
    # o braindecode descarta essas marcações; vêm direto do MOABB.
    sessions = BNCI2014_001(artifact_handling="annotate").get_data([subject_id])[subject_id]
    return {
        (session, run, round(annotation["onset"] * raw.info["sfreq"]))
        for session, runs in sessions.items()
        for run, raw in runs.items()
        for annotation in raw.annotations
        if annotation["description"] == "bnci_artifact"
    }


def _subject_row(recording):
    """Demografia do sujeito, lida do .mat pelo MOABB."""
    subject_info = recording.raw.info["subject_info"]
    return {
        "subject_id": f"A{int(recording.description['subject']):02d}",
        # idade = ano da gravação - ano de nascimento.
        "age": recording.raw.info["meas_date"].year - subject_info["birthday"].year,
        "sex": SEX[subject_info["sex"]],
    }


def build_fact_table(windows, artifact_trials) -> pd.DataFrame:
    """Fato: uma linha por trial."""
    trials = []
    for recording in windows.datasets:
        info = recording.description
        sampling_rate = recording.raw.info["sfreq"]
        subject = f"A{int(info['subject']):02d}"
        session = str(info["session"])
        run = f"run_{info['run']}"
        for _, trial in recording.metadata.iterrows():
            start = int(trial["i_start_in_trial"])
            stop = int(trial["i_stop_in_trial"])
            trial_in_run = int(trial["i_trial_in_dataset"])
            trials.append({
                "trial_id": f"{subject}_{session}_{run}_t{trial_in_run:02d}",
                "subject_id": subject,
                "session_id": session,
                "run_id": run,
                "class_id": int(trial["target"]),
                "trial_in_run": trial_in_run,
                "onset_s": start / sampling_rate,
                "duration_s": (stop - start) / sampling_rate,
                "n_samples": stop - start,
                "has_artifact": (session, str(info["run"]), start) in artifact_trials,
            })

    return pd.DataFrame(trials)


def build_dimension_tables(fact_table: pd.DataFrame, subjects):
    """Dimensões do esquema estrela."""
    dim_subject = pd.DataFrame(subjects).astype(
        {"subject_id": "string", "age": "Int16", "sex": "string"})
    # lateralidade não está no .mat -> NULL.
    dim_subject["handedness"] = pd.array([pd.NA] * len(dim_subject), dtype="string")

    dim_class = pd.DataFrame(
        [{"class_id": class_id, "class_name": class_name,
          "body_part": BODY_PART[class_name], "paradigm": "motor_imagery"}
         for class_name, class_id in CLASS_MAP.items()]
    ).sort_values("class_id").astype({
        "class_id": "int16", "class_name": "string",
        "body_part": "string", "paradigm": "string",
    })

    sessions = sorted(fact_table["session_id"].unique())
    dim_session = pd.DataFrame({
        "session_id": pd.array(sessions, dtype="string"),
        "session_role": pd.array(
            ["train" if "train" in session else "test" for session in sessions],
            dtype="string"),
    })

    runs = sorted(fact_table["run_id"].unique())
    dim_run = pd.DataFrame({
        "run_id": pd.array(runs, dtype="string"),
        "run_number": pd.array([int(run.split("_")[1]) for run in runs], dtype="int16"),
    })

    return {"dim_subject": dim_subject, "dim_class": dim_class,
            "dim_session": dim_session, "dim_run": dim_run}


def ingest(subject_ids=range(1, 10), output_dir: Path = OUTPUT_DIR):
    """Ingestão em lotes: um sujeito por vez, um row group por sujeito."""
    subject_ids = list(subject_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    fact_path = output_dir / "fact_trial.parquet"
    subjects = []
    with pq.ParquetWriter(fact_path, FACT_SCHEMA) as writer:
        for subject_id in subject_ids:
            windows = _load_windows([subject_id])
            batch = build_fact_table(windows, _artifact_trials(subject_id))
            subjects.append(_subject_row(windows.datasets[0]))
            writer.write_table(
                pa.Table.from_pandas(batch, schema=FACT_SCHEMA, preserve_index=False))

    fact_table = pd.read_parquet(fact_path)
    assert len(fact_table) == len(subject_ids) * TRIALS_PER_SUBJECT
    assert fact_table["trial_id"].is_unique

    dimensions = build_dimension_tables(fact_table, subjects)
    for name, table in dimensions.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
    # CSV da fato só para o benchmark CSV vs Parquet.
    fact_table.to_csv(output_dir / "fact_trial.csv", index=False)
    return {"fact_trial": fact_table, **dimensions}


def main():
    tables = ingest()
    for name, table in tables.items():
        print(f"{name}: {len(table)} linhas, {len(table.columns)} colunas "
              f"-> data/processed/{name}.parquet")


if __name__ == "__main__":
    main()
