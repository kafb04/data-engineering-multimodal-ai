"""Métricas e figuras do projeto, lidas das saídas já gravadas (nada é treinado aqui).

A unidade de avaliação é a pessoa: cada métrica é calculada na pessoa de teste de cada fold e
resumida entre as 9. Grava `outputs/metricas.json` e as figuras em `figuras/`.

Uso: `python -m src.evaluate`.
"""
from __future__ import annotations

import json
import os
from itertools import combinations

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.datasets import MOABBDataset
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score

from src.ingest import CLASS_MAP, DATASET_NAME
from src.samples import ROOT, folds

DATA = ROOT / "data"
FIG_DIR = ROOT / "figuras"
OUT_DIR = ROOT / "outputs"
MODELS = {"csp_lda": "CSP + LDA", "eegnet": "EEGNet", "knn": "Vizinhos (CBraMod)"}
SILVER = ["fact_trial", "dim_subject", "dim_class", "dim_session", "dim_run"]
TABLES = {**dict.fromkeys(SILVER, "processed"), "gold_valid_trials": "gold",
          "fact_prediction": "predictions", "training_history": "predictions",
          "fact_embedding": "embeddings", "fact_neighbor": "embeddings"}
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

COMPLETENESS = " UNION ALL ".join(
    f"SELECT '{table}' AS tabela, column_name AS campo, 1 - null_percentage / 100 AS completude "
    f"FROM (SUMMARIZE {table})" for table in SILVER)


def connect():
    con = duckdb.connect()
    for name, folder in TABLES.items():
        con.sql(f"CREATE VIEW {name} AS SELECT * FROM '{(DATA / folder / name).as_posix()}.parquet'")
    return con


def holm(pvalues):
    """p ajustado por Holm: o menor p × m, o seguinte × (m − 1)..., sem deixar a ordem cair."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    steps = pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(steps))
    return adjusted


def to_json(obj):
    """Converte tipos do numpy e do pandas e arredonda floats a 4 casas."""
    if isinstance(obj, dict):
        return {str(key): to_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(value) for value in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    return obj


def section_dados(con):
    """Contagens da camada Ouro, tamanho CSV × Parquet e campos incompletos da Prata."""
    by_class = con.sql(
        "SELECT class_name, SUM(n_trials)::INT AS n FROM gold_valid_trials GROUP BY 1 ORDER BY 1").df()
    by_subject = con.sql("""
        SELECT subject_id, SUM(n_trials)::INT AS n, SUM(n_trials - n_valid)::INT AS n_ruido
        FROM gold_valid_trials GROUP BY 1 ORDER BY 1""").df()
    completeness = con.sql(COMPLETENESS).df().query("completude < 1")
    csv_kib = os.path.getsize(DATA / "processed" / "fact_trial.csv") / 1024
    parquet_kib = os.path.getsize(DATA / "processed" / "fact_trial.parquet") / 1024
    return {
        "n_tentativas": by_subject["n"].sum(),
        "por_classe": dict(zip(by_class["class_name"], by_class["n"])),
        "tamanho_kib": {"csv": csv_kib, "parquet": parquet_kib, "razao": csv_kib / parquet_kib},
        "campos_incompletos": dict(zip(completeness["tabela"] + "." + completeness["campo"],
                                       completeness["completude"])),
        "tentativas_com_ruido": {
            "total": by_subject["n_ruido"].sum(),
            "fracao": by_subject["n_ruido"].sum() / by_subject["n"].sum(),
            "fracao_por_pessoa": dict(zip(by_subject["subject_id"],
                                          by_subject["n_ruido"] / by_subject["n"]))},
    }


def section_divisao(con):
    """Amostras de cada fold: só tentativas sem ruído no treino e na validação, todas no teste."""
    counts = con.sql("""
        SELECT subject_id, SUM(n_valid)::INT AS limpas, SUM(n_trials)::INT AS todas
        FROM gold_valid_trials GROUP BY 1""").df().set_index("subject_id")
    return {test: {"validacao": val, "treino": train,
                   "amostras_treino": 2 * counts.loc[train, "limpas"].sum(),
                   "amostras_validacao": 2 * counts.loc[val, "limpas"],
                   "amostras_teste": 2 * counts.loc[test, "todas"]}
            for test, val, train in folds()}


def load_predictions(con):
    """Predições dos modelos e voto dos vizinhos do CBraMod, com classe e ruído da tentativa."""
    return con.sql("""
        WITH votos AS (
            SELECT n.sample_id, e.trial_id, n.subject_id, n."window",
                   (n."window" = 'intencao')::INT AS y_true, 'knn' AS model,
                   AVG((n.neighbor_window = 'intencao')::INT) AS p_intencao
            FROM fact_neighbor n JOIN fact_embedding e USING (sample_id)
            WHERE n.pool = 'treino'
            GROUP BY ALL
        )
        SELECT p.*, c.class_name, t.has_artifact
        FROM (SELECT sample_id, trial_id, subject_id, "window", y_true, model, p_intencao, y_pred
              FROM fact_prediction
              UNION ALL BY NAME
              SELECT *, (p_intencao >= 0.5)::INT AS y_pred FROM votos) p
        JOIN fact_trial t USING (trial_id)
        JOIN dim_class c USING (class_id)
    """).df()


def per_subject(pred):
    """Área ROC, acurácia, sensibilidade e especificidade de cada modelo em cada pessoa de teste."""
    pred = pred.assign(correct=pred["y_true"] == pred["y_pred"])
    grouped = pred.groupby(["model", "subject_id"])
    table = grouped.apply(lambda g: roc_auc_score(g["y_true"], g["p_intencao"])).rename("auc").to_frame()
    table["acuracia"] = grouped["correct"].mean()
    by_class = pred.groupby(["model", "subject_id", "y_true"])["correct"].mean().unstack("y_true")
    table["sensibilidade"], table["especificidade"] = by_class[1], by_class[0]
    return table


def section_modelos(scores, history, pred):
    auc = scores["auc"].unstack("model")
    tests = [(f"{model} > acaso", stats.wilcoxon(auc[model] - 0.5, alternative="greater").pvalue) for model in MODELS]
    tests += [(f"{a} × {b}", stats.wilcoxon(auc[a], auc[b]).pvalue)
              for a, b in combinations(MODELS, 2)]
    adjusted = holm([p for _, p in tests])

    best_val = history.loc[history.groupby("test_subject")["val_auc"].idxmax()].set_index("test_subject")
    by_class = (pred[pred["y_true"] == 1].assign(correct=lambda d: d["y_pred"] == 1)
                .groupby(["model", "subject_id", "class_name"])["correct"].mean()
                .groupby(level=["model", "class_name"]).mean().unstack("class_name"))
    by_artifact = (pred.assign(correct=pred["y_true"] == pred["y_pred"])
                   .groupby(["model", "subject_id", "has_artifact"])["correct"].mean()
                   .groupby(level=["model", "has_artifact"]).mean().unstack("has_artifact")
                   .rename(columns={False: "sem_ruido", True: "com_ruido"}))
    summary = scores.groupby("model").agg(["mean", "std", "min", "max"])
    return {
        "resumo_entre_pessoas": {model: {metric: summary.loc[model, metric].to_dict()
                                         for metric in scores.columns} for model in MODELS},
        "auc_por_pessoa": auc.to_dict("index"),
        "pessoas_acima_do_acaso": (auc > 0.5).sum().to_dict(),
        "testes_wilcoxon": {name: {"p": p, "p_holm": p_holm}
                            for (name, p), p_holm in zip(tests, adjusted)},
        "eegnet_validacao_x_teste": {
            "auc_validacao_media": best_val["val_auc"].mean(),
            "auc_teste_media": auc["eegnet"].mean(),
            "epoca_escolhida": best_val["epoch"].to_dict()},
        "intencoes_detectadas_por_classe": by_class.to_dict("index"),
        "acuracia_por_ruido": by_artifact.to_dict("index"),
    }


def section_embeddings(emb, nb):
    """Vizinhos da mesma pessoa e da mesma tentativa contam fora da outra metade da tentativa."""
    everyone, train = nb[nb["pool"] == "todos"], nb[nb["pool"] == "treino"]
    other_trial = everyone[~everyone["same_trial"]]
    counts = emb["subject_id"].value_counts()
    return {
        "modelo": emb["model_version"].iat[0],
        "vizinho_da_mesma_pessoa": {
            "obtido": other_trial["same_subject"].mean(),
            "acaso": (counts * (counts - 2)).sum() / (len(emb) * (len(emb) - 2)),
            "por_pessoa": other_trial.groupby("subject_id")["same_subject"].mean().to_dict()},
        "vizinho_da_mesma_classe": {"obtido": (everyone["window"] == everyone["neighbor_window"]).mean(),
                                    "acaso": 0.5},
        "primeiro_vizinho_e_a_outra_metade_da_tentativa":
            everyone.loc[everyone["rank"] == 1, "same_trial"].mean(),
        "maior_cosseno_entre_tentativas_diferentes": other_trial["cosine_sim"].max(),
        "pares_acima_de_0_99": int((other_trial["cosine_sim"] > 0.99).sum()),
        "sinais_repetidos_por_hash": int(emb["signal_sha1"].duplicated().sum()),
        "maior_cosseno_com_outra_pessoa": train["cosine_sim"].max(),
    }


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_sinal():
    mne.viz.set_browser_backend("matplotlib")
    raw = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[1]).datasets[0].raw
    fig = raw.copy().pick(["C3", "Cz", "C4"]).plot(duration=8, start=20, scalings={"eeg": 50e-6},
                                                   show=False, show_scrollbars=False)
    fig.suptitle("A01, eletrodos C3, Cz e C4", y=1.01)
    save(fig, "sinal_c3_cz_c4.png")


def fig_classes(dados):
    counts = pd.Series(dados["por_classe"]).reindex(list(CLASS_MAP))
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(counts.index, counts.to_numpy(), color=PALETTE)
    ax.set(title="Tentativas por classe", ylabel="tentativas")
    save(fig, "distribuicao_classes.png")


def fig_treino(history):
    mean = history.groupby("epoch")[["train_loss", "val_loss", "val_auc"]].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(mean.index, mean["train_loss"], label="treino")
    left.plot(mean.index, mean["val_loss"], label="validação")
    left.set(title="Perda do EEGNet (média dos 9 folds)", xlabel="época", ylabel="entropia cruzada")
    left.legend()
    per_fold = history.pivot(index="epoch", columns="test_subject", values="val_auc")
    right.plot(per_fold, color="lightgray", lw=1)
    right.plot(mean.index, mean["val_auc"], color=PALETTE[1], lw=2.5, label="média")
    right.axhline(0.5, ls="--", c="gray", label="acaso")
    right.set(title="Área ROC na validação, por fold", xlabel="época", ylabel="área ROC")
    right.legend()
    fig.tight_layout()
    save(fig, "curvas_treino.png")


def fig_auc_por_pessoa(scores):
    auc = scores["auc"].unstack("model")[list(MODELS)]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.27
    x = np.arange(len(auc))
    for offset, (model, color) in enumerate(zip(MODELS, PALETTE)):
        ax.bar(x + (offset - 1) * width, auc[model], width, color=color,
               label=f"{MODELS[model]} (média {auc[model].mean():.2f})".replace(".", ","))
    ax.axhline(0.5, ls="--", c="gray", label="acaso")
    ax.set_xticks(x, auc.index)
    ax.set(ylim=(0.3, 1.12), yticks=np.arange(0.3, 1.01, 0.1), ylabel="área ROC no teste", xlabel="pessoa deixada de fora",
           title="Cada pessoa avaliada por modelos que nunca viram o seu sinal")
    ax.legend(ncols=2, fontsize=9)
    save(fig, "auc_por_pessoa.png")


def fig_confusao(pred):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model in zip(axes, MODELS):
        g = pred[pred["model"] == model]
        ConfusionMatrixDisplay.from_predictions(
            g["y_true"], g["y_pred"], labels=[1, 0], display_labels=["intenção", "repouso"],
            normalize="true", values_format=".0%", cmap="Blues", colorbar=False, ax=ax)
        ax.set(title=MODELS[model], xlabel="resposta do modelo", ylabel="verdade")
    fig.suptitle("Respostas nas 9 pessoas de teste juntas (corte 0,5)")
    fig.tight_layout()
    save(fig, "matriz_confusao.png")


def fig_tsne(emb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, column, title in zip(axes, ["subject_id", "window"], ["cor = pessoa", "cor = trecho"]):
        codes, labels = pd.factorize(emb[column], sort=True)
        points = ax.scatter(emb["tsne_x"], emb["tsne_y"], c=codes, cmap="tab10", s=2, alpha=0.5)
        labels = [{"intencao": "intenção"}.get(label, label) for label in labels]
        ax.legend(points.legend_elements()[0], labels, markerscale=1.5, fontsize=8)
        ax.set(title=f"t-SNE dos embeddings do CBraMod ({title})", xticks=[], yticks=[])
    fig.tight_layout()
    save(fig, "tsne.png")


def fig_vizinhos(nb):
    first = nb[~nb["same_trial"]].drop_duplicates(["pool", "sample_id"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for pool, label in [("todos", "entre todas as pessoas"), ("treino", "só entre outras pessoas")]:
        ax.hist(first.loc[first["pool"] == pool, "cosine_sim"], bins=60, alpha=0.6, label=label)
    ax.axvline(0.99, color="gray", ls="--", lw=1, label="cópia (0,99)")
    ax.set(xlabel="cosseno com o vizinho mais parecido", ylabel="trechos")
    ax.legend()
    save(fig, "semelhanca_vizinhos.png")


def main():
    mne.set_log_level("WARNING")
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    con = connect()
    pred = load_predictions(con)
    history = con.sql("SELECT * FROM training_history").df()
    scores = per_subject(pred)
    emb = con.sql("SELECT * FROM fact_embedding").df()
    nb = con.sql("SELECT * FROM fact_neighbor ORDER BY pool, sample_id, rank").df()
    metrics = {"dados": section_dados(con), "divisao": section_divisao(con),
               "modelos": section_modelos(scores, history, pred),
               "embeddings": section_embeddings(emb, nb)}
    (OUT_DIR / "metricas.json").write_text(
        json.dumps(to_json(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig_sinal()
    fig_classes(metrics["dados"])
    fig_treino(history)
    fig_auc_por_pessoa(scores)
    fig_confusao(pred)
    fig_tsne(emb)
    fig_vizinhos(nb)
    auc = scores["auc"].groupby("model").mean()
    print("área ROC média entre pessoas: " + ", ".join(f"{MODELS[m]} {auc[m]:.3f}" for m in MODELS))
    print("-> outputs/metricas.json e 7 figuras em figuras/", flush=True)


if __name__ == "__main__":
    main()
