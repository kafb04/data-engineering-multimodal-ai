"""Ingestão do BCI IV 2a em tabelas do esquema estrela (Parquet).

Fato = um trial por linha; dimensões derivadas dos metadados do MOABB, sem sinal bruto.
Uso: `python -m src.ingest`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

DATASET_NAME = "BNCI2014_001"
CLASS_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
BODY_PART = {"left_hand": "hand_left", "right_hand": "hand_right",
             "feet": "feet", "tongue": "tongue"}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_windows(subject_ids):
    # Janela de 4 s (offset 0); só para extrair metadados.
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(subject_ids))
    return create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=False, mapping=CLASS_MAP,
    )


def build_fact_table(subject_ids=range(1, 10)) -> pd.DataFrame:
    """Fato: uma linha por trial."""
    windows = _load_windows(subject_ids)
    trials = []
    for recording in windows.datasets:  # cada recording = sujeito × sessão × run
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
            })

    return pd.DataFrame(trials).astype({
        "trial_id": "string", "subject_id": "string", "session_id": "string",
        "run_id": "string", "class_id": "int16", "trial_in_run": "int16",
        "onset_s": "float64", "duration_s": "float64", "n_samples": "int32",
    })


def build_dimension_tables(fact_table: pd.DataFrame):
    """Dimensões do esquema estrela."""
    subjects = sorted(fact_table["subject_id"].unique())
    n_subjects = len(subjects)
    # 2a não publica demografia -> NULL.
    dim_subject = pd.DataFrame({
        "subject_id": pd.array(subjects, dtype="string"),
        "age": pd.array([pd.NA] * n_subjects, dtype="Int16"),
        "sex": pd.array([pd.NA] * n_subjects, dtype="string"),
        "handedness": pd.array([pd.NA] * n_subjects, dtype="string"),
    })

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
        # data de gravação não divulgada -> NULL.
        "recording_day": pd.array([pd.NaT] * len(sessions), dtype="datetime64[ns]"),
    })

    runs = sorted(fact_table["run_id"].unique())
    dim_run = pd.DataFrame({
        "run_id": pd.array(runs, dtype="string"),
        "run_number": pd.array([int(run.split("_")[1]) for run in runs], dtype="int16"),
    })

    return {"dim_subject": dim_subject, "dim_class": dim_class,
            "dim_session": dim_session, "dim_run": dim_run}


def build_tables(subject_ids=range(1, 10)):
    fact_table = build_fact_table(subject_ids)
    return {"fact_trial": fact_table, **build_dimension_tables(fact_table)}


def write_tables(tables, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
    # CSV da fato para o benchmark.
    tables["fact_trial"].to_csv(output_dir / "fact_trial.csv", index=False)


def main():
    tables = build_tables()
    write_tables(tables)
    for name, table in tables.items():
        print(f"{name}: {len(table)} linhas, {len(table.columns)} colunas "
              f"-> data/processed/{name}.parquet")


if __name__ == "__main__":
    main()
