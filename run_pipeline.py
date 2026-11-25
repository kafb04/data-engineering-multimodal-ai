"""Orquestrador do pipeline: executa as etapas do DAG em ordem e registra cada execução.

Uso:
    uv run python run_pipeline.py                     # todas as etapas
    uv run python run_pipeline.py treino avaliacao    # só as etapas nomeadas

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
from itertools import chain
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
REGISTROS = RAIZ / "outputs" / "execucoes"
SEMENTE = 42  # a mesma de src.samples.SEED, repetida para o orquestrador não importar o torch
MOABB = "MOABB BNCI2014_001 (baixado na 1ª execução para ~/mne_data)"
CBRAMOD = "Hugging Face braindecode/cbramod-pretrained (commit fixo em src/embed.py)"


def dbt(*args):
    return ["-m", "dbt.cli.main", *args, "--project-dir", "dbt", "--profiles-dir", "dbt"]


ETAPAS = {
    "ingestao": {
        "comando": ["-m", "src.ingest"],
        "depende_de": [],
        "entradas": ["src/ingest.py"],
        "externas": [MOABB],
        "saidas": ["data/processed"],
    },
    "qualidade": {
        "comando": dbt("build", "--exclude", "source:predictions"),
        "depende_de": ["ingestao"],
        "entradas": ["dbt/models", "dbt/tests", "data/processed"],
        "saidas": ["data/gold"],
    },
    "treino": {
        "comando": ["-m", "src.train"],
        "depende_de": ["qualidade"],
        "entradas": ["src/train.py", "src/samples.py", "src/ingest.py",
                     "data/processed/fact_trial.parquet"],
        "externas": [MOABB],
        "saidas": ["models", "data/predictions"],
        "semente": SEMENTE,
    },
    "testes_predicoes": {
        "comando": dbt("test", "--select", "source:predictions"),
        "depende_de": ["treino"],
        "entradas": ["dbt/models", "data/processed/fact_trial.parquet", "data/predictions"],
        "saidas": [],
    },
    "embeddings": {
        "comando": ["-m", "src.embed"],
        "depende_de": ["qualidade"],
        "entradas": ["src/embed.py", "src/samples.py", "src/ingest.py",
                     "data/processed/fact_trial.parquet"],
        "externas": [MOABB, CBRAMOD],
        "saidas": ["data/embeddings", "figuras/pares_vizinhos.png"],
        "semente": SEMENTE,
    },
    "avaliacao": {
        "comando": ["-m", "src.evaluate"],
        "depende_de": ["testes_predicoes", "embeddings"],
        "entradas": ["src/evaluate.py", "src/samples.py", "src/ingest.py", "data/processed", "data/gold", "data/predictions",
                     "data/embeddings/fact_embedding.parquet",
                     "data/embeddings/fact_neighbor.parquet"],
        "externas": [MOABB],
        "saidas": ["outputs/metricas.json", "figuras"],
    },
}


def arquivos(caminho):
    """Arquivos de um caminho; pastas são percorridas."""
    alvo = RAIZ / caminho
    if alvo.is_dir():
        return sorted(item for item in alvo.rglob("*") if item.is_file())
    return [alvo] if alvo.is_file() else []


def sha256(caminhos):
    hashes = {}
    for arquivo in chain.from_iterable(map(arquivos, caminhos)):
        with arquivo.open("rb") as conteudo:
            digest = hashlib.file_digest(conteudo, "sha256").hexdigest()
        hashes[arquivo.relative_to(RAIZ).as_posix()] = f"sha256:{digest}"
    return hashes


def atualizada(saida, desde):
    """A saída tem ao menos um arquivo gravado depois de `desde`."""
    return any(arquivo.stat().st_mtime >= desde for arquivo in arquivos(saida))


def linha_de_comando(etapa):
    return " ".join(["python", *etapa["comando"]])


def commit():
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    codigo = ["src", "dbt", "run_pipeline.py", "pyproject.toml", "uv.lock"]
    alterado = git("status", "--porcelain", "--", *codigo)
    return git("rev-parse", "--short", "HEAD") + ("+alteracoes" if alterado else "")


def executar(nome):
    etapa = ETAPAS[nome]
    registro = {
        "etapa": nome, "status": "falhou", "inicio": datetime.now().isoformat(timespec="seconds"),
        "duracao_s": None, "comando": linha_de_comando(etapa),
        "depende_de": etapa["depende_de"], "entradas": sha256(etapa["entradas"]),
        "entradas_externas": etapa.get("externas", []), "saidas": {}, "semente": etapa.get("semente"),
        "commit": commit(), "python": platform.python_version(), "sistema": platform.platform(),
    }
    for saida in map(RAIZ.joinpath, etapa["saidas"]):
        (saida.parent if saida.suffix else saida).mkdir(parents=True, exist_ok=True)
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    inicio = time.time()
    with (
        (REGISTROS / f"{nome}.log").open("w", encoding="utf-8") as log,
        subprocess.Popen([sys.executable, *etapa["comando"]], cwd=RAIZ, env=ambiente,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace") as processo,
    ):
        for linha in processo.stdout:
            print(f"  {linha}", end="", flush=True)
            log.write(linha)
    registro["duracao_s"] = round(time.time() - inicio, 1)
    if processo.returncode == 0 and all(atualizada(saida, inicio) for saida in etapa["saidas"]):
        registro.update(status="ok", saidas=sha256(etapa["saidas"]))
    return registro


def rodar(selecionadas=()):
    """Executa as etapas pedidas (todas, se nenhuma) na ordem do DAG e devolve o status de cada uma."""
    desconhecidas = set(selecionadas) - set(ETAPAS)
    if desconhecidas:
        raise ValueError(f"etapa desconhecida: {sorted(desconhecidas)}. Opções: {list(ETAPAS)}")
    REGISTROS.mkdir(parents=True, exist_ok=True)
    grafo = {nome: etapa["depende_de"] for nome, etapa in ETAPAS.items()}
    ordem = TopologicalSorter(grafo).static_order()
    status = {}
    for nome in (n for n in ordem if not selecionadas or n in selecionadas):
        bloqueio = [d for d in ETAPAS[nome]["depende_de"] if status.get(d, "ok") != "ok"]
        if bloqueio:
            registro = {"etapa": nome, "status": "nao_rodou", "duracao_s": 0,
                        "inicio": datetime.now().isoformat(timespec="seconds"),
                        "motivo": f"depende de {bloqueio}"}
        else:
            print(f"[{nome}] {linha_de_comando(ETAPAS[nome])}", flush=True)
            registro = executar(nome)
        status[nome] = registro["status"]
        (REGISTROS / f"{nome}.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{nome}] {registro['status']} ({registro['duracao_s']} s)", flush=True)
    return status


def main():
    sys.stdout.reconfigure(errors="replace")
    try:
        status = rodar(sys.argv[1:])
    except ValueError as erro:
        sys.exit(str(erro))
    print("\n" + "\n".join(f"{nome:<17} {s}" for nome, s in status.items()))
    sys.exit(0 if all(s == "ok" for s in status.values()) else 1)


if __name__ == "__main__":
    main()
