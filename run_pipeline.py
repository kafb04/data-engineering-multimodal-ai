"""Orquestrador do pipeline: executa as etapas do DAG em ordem e registra cada execução.

Uso:
    uv run python run_pipeline.py                    # todas as etapas
    uv run python run_pipeline.py treino avaliacao   # só as etapas nomeadas

Cada etapa grava `outputs/execucoes/<etapa>.json` (status, duração, sha256 das entradas e das
saídas, semente, commit) e `<etapa>.log`. Se uma etapa falha, as que dependem dela não rodam.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from graphlib import TopologicalSorter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
REGISTROS = RAIZ / "outputs" / "execucoes"
SEMENTE = 42  # a mesma de src/train.py (SEED)
MOABB = "MOABB BNCI2014_001 (baixado na 1ª execução para ~/mne_data)"
DBT = "--project-dir dbt --profiles-dir dbt"

# DAG: etapa -> comando, dependências, entradas e saídas (caminhos relativos à raiz)
ETAPAS = {
    "ingestao": {
        "comando": "python -m src.ingest",
        "depende_de": [],
        "entradas": ["src/ingest.py"],
        "externas": [MOABB],
        "saidas": ["data/processed"],
    },
    "qualidade": {  # testes da Prata e tabela Ouro, antes de treinar
        "comando": f"python -m dbt.cli.main build {DBT} --exclude source:predictions",
        "depende_de": ["ingestao"],
        "entradas": ["dbt/models", "data/processed"],
        "saidas": ["data/gold"],
    },
    "treino": {
        "comando": "python -m src.train",
        "depende_de": ["qualidade"],
        "entradas": ["src/train.py", "data/processed/fact_trial.parquet"],
        "externas": [MOABB],
        "saidas": ["models", "data/predictions"],
        "semente": SEMENTE,
    },
    "testes_predicoes": {  # testes da tabela de predições, que só existe depois do treino
        "comando": f"python -m dbt.cli.main test {DBT} --select source:predictions",
        "depende_de": ["treino"],
        "entradas": ["dbt/models", "data/processed/fact_trial.parquet", "data/predictions"],
        "saidas": [],
    },
    "embeddings": {
        "comando": "python -m src.embed",
        "depende_de": ["treino"],
        "entradas": ["src/embed.py", "data/processed/fact_trial.parquet",
                     "data/predictions/fact_prediction.parquet"],
        "externas": [MOABB, "Hugging Face braindecode/cbramod-pretrained (commit fixo em src/embed.py)"],
        "saidas": ["data/embeddings", "figuras/pares_vizinhos.png"],
        "semente": SEMENTE,
    },
    "avaliacao": {
        "comando": "python -m src.evaluate",
        "depende_de": ["testes_predicoes", "embeddings"],
        "entradas": ["src/evaluate.py", "data/processed", "data/gold", "data/predictions",
                     "data/embeddings"],
        "saidas": ["outputs/metricas.json", "figuras"],
    },
}


def sha256(caminhos):
    """{arquivo: sha256} de cada arquivo (pastas são percorridas)."""
    hashes = {}
    for caminho in caminhos:
        alvo = RAIZ / caminho
        for arquivo in sorted(alvo.rglob("*")) if alvo.is_dir() else [alvo]:
            if arquivo.is_file():
                with arquivo.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                hashes[arquivo.relative_to(RAIZ).as_posix()] = f"sha256:{digest}"
    return hashes


def commit():
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("+alteracoes" if git("status", "--porcelain") else "")


def executar(nome):
    etapa = ETAPAS[nome]
    registro = {
        "etapa": nome, "status": "falhou", "inicio": datetime.now().isoformat(timespec="seconds"),
        "duracao_s": None, "comando": etapa["comando"], "depende_de": etapa["depende_de"],
        "entradas": sha256(etapa["entradas"]), "entradas_externas": etapa.get("externas", []),
        "saidas": {}, "semente": etapa.get("semente"), "commit": commit(),
        "python": platform.python_version(), "sistema": platform.platform(),
    }
    for saida in etapa["saidas"]:  # o dbt não cria a pasta de destino
        pasta = RAIZ / saida
        (pasta.parent if pasta.suffix else pasta).mkdir(parents=True, exist_ok=True)
    comando = [sys.executable, *etapa["comando"].split()[1:]]
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    inicio = time.time()
    with (REGISTROS / f"{nome}.log").open("w", encoding="utf-8") as log:
        processo = subprocess.Popen(comando, cwd=RAIZ, env=ambiente, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
        for linha in processo.stdout:  # mostra na tela e guarda no log
            print(f"  {linha}", end="", flush=True)
            log.write(linha)
        codigo = processo.wait()
    registro["duracao_s"] = round(time.time() - inicio, 1)
    if codigo == 0:
        registro["saidas"] = sha256(etapa["saidas"])
        faltam = [s for s in etapa["saidas"] if not (RAIZ / s).exists()]
        registro["status"] = "falhou" if faltam else "ok"
    return registro


def main(selecionadas):
    desconhecidas = set(selecionadas) - set(ETAPAS)
    if desconhecidas:
        sys.exit(f"etapa desconhecida: {sorted(desconhecidas)}. Opções: {list(ETAPAS)}")
    sys.stdout.reconfigure(errors="replace")
    REGISTROS.mkdir(parents=True, exist_ok=True)
    ordem = TopologicalSorter({n: e["depende_de"] for n, e in ETAPAS.items()}).static_order()
    status = {}
    for nome in ordem:
        if selecionadas and nome not in selecionadas:
            continue  # etapa não pedida: vale o que já está no disco
        bloqueio = [d for d in ETAPAS[nome]["depende_de"] if status.get(d, "ok") != "ok"]
        if bloqueio:
            registro = {"etapa": nome, "status": "nao_rodou",
                        "inicio": datetime.now().isoformat(timespec="seconds"), "duracao_s": 0,
                        "motivo": f"depende de {bloqueio}"}
        else:
            print(f"[{nome}] {ETAPAS[nome]['comando']}", flush=True)
            registro = executar(nome)
        status[nome] = registro["status"]
        (REGISTROS / f"{nome}.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{nome}] {registro['status']} ({registro['duracao_s']} s)", flush=True)
    print("\n" + "\n".join(f"{nome:<11} {s}" for nome, s in status.items()))
    sys.exit(0 if all(s == "ok" for s in status.values()) else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
