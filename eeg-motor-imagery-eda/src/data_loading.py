"""
data_loading.py
----------------
Funções utilitárias para baixar, carregar e padronizar registros do
PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb) usando MNE-Python.

O download usa o mecanismo oficial `mne.datasets.eegbci.load_data`, que
busca os arquivos diretamente em https://physionet.org/files/eegmmidb/1.0.0/
(acesso aberto, sem necessidade de login) e faz cache local em
`~/mne_data` (ou no diretório passado em `data_dir`).

Requer conexão com a internet na primeira execução. Funciona em
ambiente local, Google Colab e Kaggle Kernels.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
from mne.datasets import eegbci

mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Metadados do dataset (ver docs/DATASHEET.md para a descrição completa)
# ---------------------------------------------------------------------------

# Sujeitos com inconsistências conhecidas de anotação/duração de trials,
# rotineiramente excluídos na literatura (Kim et al., 2024 -
# "Increasing accessibility to a large brain-computer interface dataset").
KNOWN_PROBLEMATIC_SUBJECTS = [88, 89, 92, 100, 104, 106]

# Mapeamento run -> (tipo de tarefa, descrição, classes em T1/T2)
RUN_TASK_MAP = {
    1: ("baseline", "Linha de base - olhos abertos", None),
    2: ("baseline", "Linha de base - olhos fechados", None),
    3: ("execucao", "Execução motora real: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    4: ("imagetica", "Imagética motora: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    5: ("execucao", "Execução motora real: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
    6: ("imagetica", "Imagética motora: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
    7: ("execucao", "Execução motora real: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    8: ("imagetica", "Imagética motora: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    9: ("execucao", "Execução motora real: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
    10: ("imagetica", "Imagética motora: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
    11: ("execucao", "Execução motora real: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    12: ("imagetica", "Imagética motora: mão esquerda (T1) vs. mão direita (T2)", ("mao_esquerda", "mao_direita")),
    13: ("execucao", "Execução motora real: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
    14: ("imagetica", "Imagética motora: ambas as mãos (T1) vs. ambos os pés (T2)", ("ambas_maos", "ambos_pes")),
}

SFREQ_EXPECTED = 160.0
N_CHANNELS_EXPECTED = 64


def valid_subject_ids(subjects: list[int]) -> list[int]:
    """Remove da lista os sujeitos com inconsistências conhecidas."""
    removed = [s for s in subjects if s in KNOWN_PROBLEMATIC_SUBJECTS]
    if removed:
        warnings.warn(
            f"Sujeitos removidos por inconsistências conhecidas na literatura: {removed}"
        )
    return [s for s in subjects if s not in KNOWN_PROBLEMATIC_SUBJECTS]


def download_runs(subject: int, runs: list[int], data_dir: str | Path) -> list[str]:
    """Baixa (ou reutiliza do cache) os arquivos .edf de um sujeito/runs.

    Parameters
    ----------
    subject : int
        Número do sujeito (1-109).
    runs : list[int]
        Números das corridas (runs) desejadas (1-14).
    data_dir : str | Path
        Diretório local onde os dados serão armazenados em cache.

    Returns
    -------
    list[str]
        Caminhos locais dos arquivos .edf baixados.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = eegbci.load_data(subject, runs, path=str(data_dir), update_path=False)
    return paths


def load_raw(subject: int, runs: list[int], data_dir: str | Path) -> mne.io.Raw:
    """Baixa (se necessário), carrega, concatena e padroniza um conjunto de runs.

    Aplica `eegbci.standardize` (corrige nomes de canais para o padrão
    10-05) e define a montagem `standard_1005` para permitir análises
    espaciais (ex.: plot de sensores, CSP).
    """
    paths = download_runs(subject, runs, data_dir)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="WARNING") for p in paths]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)  # renomeia canais para o padrão 10-05 (remove sufixos)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")
    return raw


def raw_summary_row(subject: int, run: int, raw: mne.io.Raw) -> dict:
    """Extrai uma linha de resumo (metadados) de um objeto Raw carregado."""
    events, event_id = mne.events_from_annotations(raw, verbose="WARNING")
    counts = {name: int((events[:, 2] == idx).sum()) for name, idx in event_id.items()}
    task_type, description, _ = RUN_TASK_MAP.get(run, ("desconhecido", "desconhecido", None))
    return {
        "subject": f"S{subject:03d}",
        "run": f"R{run:02d}",
        "task_type": task_type,
        "description": description,
        "sfreq_hz": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "duration_s": round(raw.n_times / raw.info["sfreq"], 1),
        "n_events_total": int(events.shape[0]),
        **{f"n_{k}": v for k, v in counts.items()},
    }
