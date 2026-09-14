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

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _windows(subject_ids):
    # Janela de 4 s (offset 0); só para extrair metadados.
    ds = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=list(subject_ids))
    return create_windows_from_events(
        ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=False, mapping=CLASS_MAP,
    )


def build_fact(subject_ids=range(1, 10)) -> pd.DataFrame:
    """Fato: uma linha por trial."""
    wd = _windows(subject_ids)
    rows = []
    for d in wd.datasets:
        desc = d.description
        sfreq = d.raw.info["sfreq"]
        subject = f"A{int(desc['subject']):02d}"
        session = str(desc["session"])
        run = f"run_{desc['run']}"
        for _, r in d.metadata.iterrows():
            start, stop = int(r["i_start_in_trial"]), int(r["i_stop_in_trial"])
            trial_in_run = int(r["i_trial_in_dataset"])
            rows.append({
                "trial_id": f"{subject}_{session}_{run}_t{trial_in_run:02d}",
                "subject_id": subject,
                "session_id": session,
                "run_id": run,
                "class_id": int(r["target"]),
                "trial_in_run": trial_in_run,
                "onset_s": start / sfreq,
                "duration_s": (stop - start) / sfreq,
                "n_samples": stop - start,
            })

    return pd.DataFrame(rows).astype({
        "trial_id": "string", "subject_id": "string", "session_id": "string",
        "run_id": "string", "class_id": "int16", "trial_in_run": "int16",
        "onset_s": "float64", "duration_s": "float64", "n_samples": "int32",
    })


def build_dims(fact: pd.DataFrame):
    """Dimensões do esquema estrela."""
    subjects = sorted(fact["subject_id"].unique())
    n = len(subjects)
    # 2a não publica demografia -> NULL.
    dim_subject = pd.DataFrame({
        "subject_id": pd.array(subjects, dtype="string"),
        "age": pd.array([pd.NA] * n, dtype="Int16"),
        "sex": pd.array([pd.NA] * n, dtype="string"),
        "handedness": pd.array([pd.NA] * n, dtype="string"),
    })

    dim_class = pd.DataFrame(
        [{"class_id": c, "class_name": name, "body_part": BODY_PART[name],
          "paradigm": "motor_imagery"} for name, c in CLASS_MAP.items()]
    ).sort_values("class_id").astype({
        "class_id": "int16", "class_name": "string",
        "body_part": "string", "paradigm": "string",
    })

    sessions = sorted(fact["session_id"].unique())
    dim_session = pd.DataFrame({
        "session_id": pd.array(sessions, dtype="string"),
        "session_role": pd.array(
            ["train" if "train" in s else "test" for s in sessions], dtype="string"),
        # data de gravação não divulgada -> NULL.
        "recording_day": pd.array([pd.NaT] * len(sessions), dtype="datetime64[ns]"),
    })

    runs = sorted(fact["run_id"].unique())
    dim_run = pd.DataFrame({
        "run_id": pd.array(runs, dtype="string"),
        "run_number": pd.array([int(r.split("_")[1]) for r in runs], dtype="int16"),
    })

    return {"dim_subject": dim_subject, "dim_class": dim_class,
            "dim_session": dim_session, "dim_run": dim_run}


def build_tables(subject_ids=range(1, 10)):
    fact = build_fact(subject_ids)
    return {"fact_trial": fact, **build_dims(fact)}


def write_tables(tables, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    # CSV da fato para o benchmark.
    tables["fact_trial"].to_csv(out_dir / "fact_trial.csv", index=False)


def main():
    tables = build_tables()
    write_tables(tables)
    for name, df in tables.items():
        print(f"{name}: {len(df)} linhas, {len(df.columns)} colunas "
              f"-> data/processed/{name}.parquet")


if __name__ == "__main__":
    main()
