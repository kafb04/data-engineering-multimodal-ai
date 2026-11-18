"""Métricas e figuras do projeto, a partir das saídas já gravadas (nada é treinado nem ingerido).

Lê data/processed, data/predictions e data/embeddings (e o cache do MOABB, só para o gráfico do
sinal) e grava `outputs/metricas.json` e as figuras em `figuras/`. Substitui os notebooks 01 e 02.
Todas as proporções do JSON ficam entre 0 e 1.

Uso: `python -m src.evaluate`.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.datasets import MOABBDataset
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.ingest import CLASS_MAP, DATASET_NAME
from src.train import TEST_SUBJECTS, TRAIN_SUBJECTS, VAL_SUBJECTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "figuras"
OUT_DIR = ROOT / "outputs"
MODELS = ["eegconformer", "mlp"]
PALETTE = ["#ec4899", "#8a6fa8", "#f97316", "#22a39f"]
TABLES = ["fact_trial", "dim_subject", "dim_class", "dim_session", "dim_run"]
FIGURES = []


def connect():
    con = duckdb.connect()
    for folder, names in [("processed", TABLES), ("predictions", ["fact_prediction", "training_history"]),
                          ("embeddings", ["fact_embedding", "fact_neighbor"])]:
        for name in names:
            con.sql(f"CREATE VIEW {name} AS SELECT * FROM '{(DATA / folder / name).as_posix()}.parquet'")
    return con


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(name)


def clean(obj):
    """Tipos do numpy/pandas -> JSON; floats com 4 casas; NaN -> null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else round(float(obj), 4)
    return obj


def count_leaves(obj):
    if isinstance(obj, dict):
        return sum(count_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_leaves(v) for v in obj)
    return 1


# ---------- estatística (portada do notebook 02) ----------

def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    ci = stats.binomtest(int(k), int(n)).proportion_ci(method="wilson")
    return k / n, ci.low, ci.high


def wilson_dict(k, n):
    value, low, high = wilson(k, n)
    return {"valor": value, "ic_inf": low, "ic_sup": high, "k": int(k), "n": int(n)}


def confusion_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"sensibilidade": (tp, tp + fn), "especificidade": (tn, tn + fp),
            "precisão": (tp, tp + fp), "acurácia": (tp + tn, tp + fn + fp + tn)}


def delong(y_true, score_a, score_b):
    """Teste de DeLong para duas áreas sob a curva ROC nas mesmas amostras."""
    pos, neg = y_true == 1, y_true == 0
    aucs, v10, v01 = [], [], []
    for score in (score_a, score_b):
        # comparação de cada positivo com cada negativo: 1 se o positivo tem saída maior, 0,5 se empata
        psi = (score[pos][:, None] > score[neg][None, :]) + 0.5 * (score[pos][:, None] == score[neg][None, :])
        aucs.append(psi.mean())
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
    s10, s01 = np.cov(v10), np.cov(v01)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / pos.sum() + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / neg.sum()
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, 2 * stats.norm.sf(abs(z))


def compare(df, column):
    table = pd.crosstab(df[column], df["correct"]).to_numpy()
    if table.shape != (table.shape[0], 2):
        return np.nan, "não aplicável (todos certos ou todos errados)"
    if table.shape == (2, 2) and (stats.contingency.expected_freq(table) < 5).any():
        return stats.fisher_exact(table)[1], "Fisher"
    return stats.chi2_contingency(table)[1], "qui-quadrado"


def holm(pvalues):
    # multiplica o menor p por m, o seguinte por m - 1 e assim por diante, sem deixar a ordem cair
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvalues[i] * (len(pvalues) - rank)))
        adjusted[i] = running
    return adjusted


# ---------- seções do JSON ----------

def section_dados(con):
    fact = con.sql("SELECT * FROM fact_trial").df()
    por_classe = con.sql("""
        SELECT c.class_name, COUNT(*) AS n FROM fact_trial f JOIN dim_class c USING (class_id)
        GROUP BY 1 ORDER BY 1""").df().set_index("class_name")["n"].to_dict()
    por_papel = con.sql("""
        SELECT s.session_role, c.class_name, COUNT(*) AS n
        FROM fact_trial f JOIN dim_class c USING (class_id) JOIN dim_session s USING (session_id)
        GROUP BY 1, 2 ORDER BY 1, 2""").df()
    grupo = con.sql("""
        SELECT MIN(n_trials) AS min_por_classe_run, MAX(n_trials) AS max_por_classe_run, COUNT(*) AS n_grupos
        FROM (SELECT COUNT(*) AS n_trials FROM fact_trial GROUP BY subject_id, session_id, run_id, class_id)
    """).df().iloc[0].to_dict()
    gaps = con.sql("""
        WITH gaps AS (
            SELECT session_id, onset_s, duration_s,
                   onset_s - LAG(onset_s) OVER (
                       PARTITION BY subject_id, session_id, run_id ORDER BY onset_s) AS gap_s
            FROM fact_trial)
        SELECT s.session_role, AVG(gap_s) AS gap_medio_s, MIN(gap_s) AS gap_min_s,
               MAX(gap_s) AS gap_max_s, MAX(onset_s + duration_s) AS fim_ultimo_trial_s
        FROM gaps JOIN dim_session s USING (session_id) GROUP BY 1 ORDER BY 1""").df()
    csv_kib = os.path.getsize(DATA / "processed" / "fact_trial.csv") / 1024
    parquet_kib = os.path.getsize(DATA / "processed" / "fact_trial.parquet") / 1024
    return {
        "n_tentativas": len(fact),
        "por_classe": por_classe,
        "por_papel_de_sessao_e_classe": {
            role: g.set_index("class_name")["n"].to_dict() for role, g in por_papel.groupby("session_role")},
        "trials_por_run_e_classe": grupo,
        "intervalo_entre_trials": gaps.set_index("session_role").to_dict("index"),
        "tamanho": {"csv_kib": csv_kib, "parquet_kib": parquet_kib, "razao_csv_parquet": csv_kib / parquet_kib},
    }


RULES = """
    WITH seq AS (
        SELECT onset_s - LAG(onset_s) OVER (
                   PARTITION BY subject_id, session_id, run_id ORDER BY trial_in_run) AS gap_s
        FROM fact_trial
    )
    SELECT 'class_id fora de 0–3' AS regra, COUNT(*) AS violacoes
    FROM fact_trial WHERE class_id NOT BETWEEN 0 AND 3
    UNION ALL
    SELECT 'trial_in_run fora de 0–47', COUNT(*)
    FROM fact_trial WHERE trial_in_run NOT BETWEEN 0 AND 47
    UNION ALL
    SELECT 'duration_s diferente de 4,0', COUNT(*) FROM fact_trial WHERE duration_s <> 4.0
    UNION ALL
    SELECT 'n_samples diferente de duration_s × 250', COUNT(*)
    FROM fact_trial WHERE n_samples <> duration_s * 250
    UNION ALL
    SELECT 'onset_s negativo', COUNT(*) FROM fact_trial WHERE onset_s < 0
    UNION ALL
    SELECT 'sentinela (-1 ou 999) em campo numérico', COUNT(*)
    FROM fact_trial
    WHERE class_id IN (-1, 999) OR trial_in_run IN (-1, 999) OR n_samples IN (-1, 999)
       OR onset_s IN (-1, 999) OR duration_s IN (-1, 999)
    UNION ALL
    SELECT 'subject_id fora do padrão A01–A09', COUNT(*)
    FROM fact_trial WHERE NOT regexp_full_match(subject_id, 'A0[1-9]')
    UNION ALL
    SELECT 'trial_id diferente da composição das chaves', COUNT(*)
    FROM fact_trial
    WHERE trial_id <> subject_id || '_' || session_id || '_' || run_id
                      || '_t' || lpad(trial_in_run::VARCHAR, 2, '0')
    UNION ALL
    SELECT 'subject_id sem linha em dim_subject', COUNT(*)
    FROM fact_trial ANTI JOIN dim_subject USING (subject_id)
    UNION ALL
    SELECT 'session_id sem linha em dim_session', COUNT(*)
    FROM fact_trial ANTI JOIN dim_session USING (session_id)
    UNION ALL
    SELECT 'run_id sem linha em dim_run', COUNT(*)
    FROM fact_trial ANTI JOIN dim_run USING (run_id)
    UNION ALL
    SELECT 'class_id sem linha em dim_class', COUNT(*)
    FROM fact_trial ANTI JOIN dim_class USING (class_id)
    UNION ALL
    SELECT 'run com nº de trials diferente de 48', COUNT(*)
    FROM (SELECT COUNT(*) AS n FROM fact_trial GROUP BY subject_id, session_id, run_id)
    WHERE n <> 48
    UNION ALL
    SELECT 'run × classe com nº de trials diferente de 12', COUNT(*)
    FROM (SELECT COUNT(*) AS n FROM fact_trial GROUP BY subject_id, session_id, run_id, class_id)
    WHERE n <> 12
    UNION ALL
    SELECT 'onset_s fora de ordem dentro do run', COUNT(*) FROM seq WHERE gap_s <= 0
    UNION ALL
    SELECT 'intervalo entre trials fora de 7,5–8,5 s', COUNT(*)
    FROM seq WHERE gap_s NOT BETWEEN 7.5 AND 8.5
    UNION ALL
    SELECT 'dim_subject.age fora de 10–100', COUNT(*)
    FROM dim_subject WHERE age NOT BETWEEN 10 AND 100
    UNION ALL
    SELECT 'dim_subject.sex fora de {M, F}', COUNT(*)
    FROM dim_subject WHERE sex NOT IN ('M', 'F')
"""

DUPLICATES = """
    SELECT 'fact_trial.trial_id' AS chave, COUNT(*) AS linhas,
           COUNT(*) - COUNT(DISTINCT trial_id) AS duplicatas
    FROM fact_trial
    UNION ALL
    SELECT 'fact_trial (sujeito, sessão, run, trial_in_run)', COUNT(*),
           COUNT(*) - COUNT(DISTINCT (subject_id, session_id, run_id, trial_in_run))
    FROM fact_trial
    UNION ALL
    SELECT 'fact_trial (sujeito, sessão, run, onset_s)', COUNT(*),
           COUNT(*) - COUNT(DISTINCT (subject_id, session_id, run_id, onset_s))
    FROM fact_trial
    UNION ALL
    SELECT 'dim_subject.subject_id', COUNT(*), COUNT(*) - COUNT(DISTINCT subject_id) FROM dim_subject
    UNION ALL
    SELECT 'dim_class.class_id', COUNT(*), COUNT(*) - COUNT(DISTINCT class_id) FROM dim_class
    UNION ALL
    SELECT 'dim_session.session_id', COUNT(*), COUNT(*) - COUNT(DISTINCT session_id) FROM dim_session
    UNION ALL
    SELECT 'dim_run.run_id', COUNT(*), COUNT(*) - COUNT(DISTINCT run_id) FROM dim_run
"""


def section_qualidade(con):
    dup = con.sql(DUPLICATES).df().set_index("chave")
    completude = {}
    for table in TABLES:
        s = con.sql(f"SELECT column_name AS campo, 1 - null_percentage / 100 AS completude "
                    f"FROM (SUMMARIZE {table})").df()
        completude[table] = {r.campo: float(r.completude) for r in s.itertuples()}
    regras = con.sql(RULES).df().set_index("regra")["violacoes"].to_dict()
    art = con.sql("""
        SELECT COALESCE(subject_id, 'total') AS subject_id,
               COUNT(*) FILTER (has_artifact) AS n_artefato, COUNT(*) AS n_trials,
               COUNT(*) FILTER (has_artifact) / COUNT(*) AS fracao_artefato
        FROM fact_trial GROUP BY ROLLUP (subject_id) ORDER BY GROUPING(subject_id), subject_id""").df()
    por_classe = con.sql("""
        SELECT c.class_name, SUM(n_artefato)::INT AS n_artefato,
               MIN(restantes) AS min_restantes, MAX(restantes) AS max_restantes
        FROM (SELECT subject_id, session_id, class_id, COUNT(*) FILTER (has_artifact) AS n_artefato,
                     COUNT(*) FILTER (NOT has_artifact) AS restantes
              FROM fact_trial GROUP BY ALL)
        JOIN dim_class c USING (class_id) GROUP BY 1 ORDER BY 1""").df()
    por_sujeito = art.set_index("subject_id").to_dict("index")
    return {
        "duplicatas_por_chave": dup.to_dict("index"),
        "completude_por_campo": completude,
        "violacoes_por_regra": regras,
        "n_regras": len(regras),
        "artefatos": {"total": por_sujeito.pop("total"), "por_sujeito": por_sujeito,
                      "por_classe": por_classe.set_index("class_name").to_dict("index")},
    }


def section_pipeline(con):
    return con.sql(f"""
        WITH t AS (
            SELECT CASE WHEN subject_id IN {tuple(TRAIN_SUBJECTS)} THEN '1. treino'
                        WHEN subject_id IN {tuple(VAL_SUBJECTS)} THEN '2. validação'
                        ELSE '3. teste' END AS conjunto, has_artifact
            FROM fact_trial)
        SELECT conjunto, COUNT(*) AS trials, SUM(has_artifact::INT)::INT AS com_artefato,
               CASE WHEN conjunto = '3. teste' THEN COUNT(*)
                    ELSE COUNT(*) - SUM(has_artifact::INT)::INT END AS trials_usados,
               2 * trials_usados AS amostras
        FROM t GROUP BY conjunto ORDER BY conjunto""").df().set_index("conjunto").to_dict("index")


def load_predictions(con):
    # uma linha por predição, com o membro imaginado, o artefato e a demografia vindos das dimensões
    pred = con.sql("""
        SELECT p.*, c.class_name, t.has_artifact, s.age, s.sex
        FROM fact_prediction p
        JOIN fact_trial t USING (trial_id)
        JOIN dim_class c USING (class_id)
        JOIN dim_subject s ON s.subject_id = p.subject_id""").df()
    pred["correct"] = pred["y_true"] == pred["y_pred"]
    return pred


def section_modelos(pred):
    test, val = pred[pred["split"] == "teste"], pred[pred["split"] == "validacao"]
    out = {}
    for model in MODELS:
        g, v = test[test["model"] == model], val[val["model"] == model]
        threshold = float(g["threshold"].iat[0])
        cortes = {}
        for label, cut in [("0,50", 0.5), ("validação", threshold)]:
            y_pred = (g["p_intencao"] >= cut).astype(int)
            cortes[label] = {"limiar": cut, "fracao_intencao": y_pred.mean(),
                             **{m: wilson_dict(k, n) for m, (k, n) in confusion_metrics(g["y_true"], y_pred).items()}}
        subjects = {}
        for subject, s in g.groupby("subject_id"):
            mi, rest = s[s["y_true"] == 1], s[s["y_true"] == 0]
            subjects[subject] = {
                "auc": roc_auc_score(s["y_true"], s["p_intencao"]), "idade": s["age"].iat[0], "sexo": s["sex"].iat[0],
                "acurácia": wilson_dict(s["correct"].sum(), len(s)),
                "sensibilidade": wilson_dict(mi["correct"].sum(), len(mi)),
                "especificidade": wilson_dict(rest["correct"].sum(), len(rest)),
                "fracao_intencao": s["y_pred"].mean()}
        out[model] = {"auc_validacao": roc_auc_score(v["y_true"], v["p_intencao"]),
                      "auc_teste": roc_auc_score(g["y_true"], g["p_intencao"]),
                      "limiar_validacao": threshold, "n_validacao": len(v), "n_teste": len(g),
                      "cortes": cortes, "por_sujeito_de_teste": subjects}
    return out


def section_comparacoes(pred, chosen):
    test = pred[pred["split"] == "teste"]
    paired = test.pivot(index="sample_id", columns="model", values="correct")
    only_conformer = int((paired["eegconformer"] & ~paired["mlp"]).sum())
    only_mlp = int((~paired["eegconformer"] & paired["mlp"]).sum())
    p_mcnemar = stats.binomtest(only_conformer, only_conformer + only_mlp).pvalue

    scores = test.pivot(index="sample_id", columns="model", values="p_intencao")
    labels = test.drop_duplicates("sample_id").set_index("sample_id").loc[scores.index, "y_true"].to_numpy()
    (auc_c, auc_m), p_delong = delong(labels, scores["eegconformer"].to_numpy(), scores["mlp"].to_numpy())

    best = test[test["model"] == chosen]
    groups = {"class_name": best[best["y_true"] == 1], "has_artifact": best, "subject_id": best}
    labels_ = {"class_name": "o membro imaginado", "has_artifact": "o artefato", "subject_id": "o sujeito"}
    group_tests = {column: compare(df, column) for column, df in groups.items()}
    tests = [("mcnemar", p_mcnemar, "McNemar"), ("delong", p_delong, "DeLong"),
             *[(column, p, name) for column, (p, name) in group_tests.items()]]
    adjusted = holm(np.array([t[1] for t in tests]))
    result = {"modelo_escolhido": chosen,
              "mcnemar": {"so_eegconformer_acerta": only_conformer, "so_mlp_acerta": only_mlp},
              "delong": {"auc_eegconformer": auc_c, "auc_mlp": auc_m}}
    for (key, p, name), p_holm in zip(tests, adjusted):
        item = result.setdefault(key, {"pergunta": labels_.get(key)} if key in labels_ else {})
        item.update({"teste": name, "p": p, "p_holm": p_holm})
    for column, df in groups.items():
        result[column]["grupos"] = df.groupby(column).agg(
            pacientes=("subject_id", "nunique"), amostras=("correct", "size"),
            acerto=("correct", "mean")).to_dict("index")
    return result


def section_embeddings(con, pred):
    emb = con.sql("SELECT * FROM fact_embedding").df()
    neighbor = con.sql("SELECT * FROM fact_neighbor").df()
    n_total = len(emb)
    pool_all = neighbor[neighbor["pool"] == "todos"]
    by_subject = pool_all.groupby("subject_id").agg(mesmo_sujeito=("same_subject", "mean"))
    by_subject["acaso"] = (emb["subject_id"].value_counts() - 1) / (n_total - 1)
    by_subject.loc["todos"] = [pool_all["same_subject"].mean(), (n_total / 9 - 1) / (n_total - 1)]

    top1 = neighbor[(neighbor["rank"] == 1) & (neighbor["split"] == "teste")]
    pairs = top1[top1["pool"] == "treino"].sort_values("cosine_sim", ascending=False)
    cos = {pool: g["cosine_sim"].describe().drop("count").to_dict() for pool, g in top1.groupby("pool")}

    pool_train = neighbor[neighbor["pool"] == "treino"]
    knn = (pool_train.assign(vote=pool_train["neighbor_window"] == "intencao")
           .groupby(["sample_id", "subject_id", "split", "window"], as_index=False)["vote"].mean())
    knn["y_true"] = (knn["window"] == "intencao").astype(int)
    knn_split = {}
    for split, g in knn.groupby("split"):
        y_pred = (g["vote"] > 0.5).astype(int)
        knn_split[split] = {
            "auc": roc_auc_score(g["y_true"], g["vote"]),
            "sensibilidade": wilson_dict(((y_pred == 1) & (g["y_true"] == 1)).sum(), (g["y_true"] == 1).sum()),
            "especificidade": wilson_dict(((y_pred == 0) & (g["y_true"] == 0)).sum(), (g["y_true"] == 0).sum())}
    knn_test = knn[knn["split"] == "teste"]
    return {
        "modelo": emb["model_version"].iat[0], "n_amostras": n_total, "n_pares_vizinhos": len(neighbor),
        "mesma_pessoa": by_subject.to_dict("index"),
        "primeiro_vizinho_outra_janela_do_mesmo_trial": pool_all.loc[pool_all["rank"] == 1, "same_trial"].mean(),
        "mesma_classe": (pool_all["window"] == pool_all["neighbor_window"]).mean(),
        "primeiro_vizinho_do_teste_mesma_pessoa_em_todas": top1.loc[top1["pool"] == "todos", "same_subject"].mean(),
        "cosseno_primeiro_vizinho_teste": cos,
        "maior_cosseno_teste_treino": pairs["cosine_sim"].max(),
        "pares_teste_treino_acima_de_0_99": int((pairs["cosine_sim"] > 0.99).sum()),
        "hashes_repetidos": int(emb["signal_sha1"].duplicated().sum()),
        "pares_mais_parecidos": pairs[["sample_id", "neighbor_sample_id", "cosine_sim"]].head(5).to_dict("records"),
        "knn": {"por_conjunto": knn_split,
                "auc_teste_por_sujeito": knn_test.groupby("subject_id")[["vote", "y_true"]].apply(
                    lambda g: roc_auc_score(g["y_true"], g["vote"])).to_dict()},
        "auc_modelos_por_conjunto": {model: {split: roc_auc_score(g["y_true"], g["p_intencao"])
                                             for split, g in pred[pred["model"] == model].groupby("split")}
                                     for model in MODELS},
    }, emb, top1


# ---------- figuras ----------

def fig_sinal():
    mne.set_log_level("WARNING")
    mne.viz.set_browser_backend("matplotlib")
    dataset = MOABBDataset(dataset_name=DATASET_NAME, subject_ids=[1])
    raw = dataset.datasets[0].raw
    motor = [ch for ch in ["C3", "Cz", "C4"] if ch in raw.ch_names]
    fig = raw.copy().pick(motor).plot(duration=8, start=20, scalings=dict(eeg=50e-6),
                                      show=False, show_scrollbars=False)
    fig.suptitle(f"A01 — canais {motor}", y=1.01)
    save(fig, "sinal_c3_cz_c4.png")


def fig_classes(con):
    counts = con.sql("""SELECT c.class_name, COUNT(*) AS n FROM fact_trial JOIN dim_class c USING (class_id)
                        GROUP BY 1""").df().set_index("class_name")["n"].reindex(list(CLASS_MAP))
    fig, ax = plt.subplots(figsize=(5.5, 3))
    counts.plot(kind="bar", color=PALETTE, rot=0, title="Distribuição de classes", ax=ax)
    ax.set_xlabel("")
    save(fig, "distribuicao_classes.png")


def fig_treino(con):
    history = con.sql("SELECT * FROM training_history ORDER BY model, epoch").df()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (model, h) in zip(axes, history.groupby("model")):
        best = h.loc[h["val_auc"].idxmax(), "epoch"]
        ax.plot(h["epoch"], h["train_loss"], label="perda no treino")
        ax.plot(h["epoch"], h["val_loss"], label="perda na validação")
        ax.plot(h["epoch"], h["val_auc"], label="área sob a curva ROC na validação")
        ax.axvline(best, ls="--", c="gray", label=f"época escolhida ({best})")
        ax.set(title=model, xlabel="época")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "curvas_treino.png")


def fig_confusao(test):
    names = {"eegconformer": "EEGConformer", "mlp": "MLP"}
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for row, model in zip(axes, MODELS):
        g = test[test["model"] == model]
        for ax, (label, threshold) in zip(row, [("padrão", 0.5), ("da validação", g["threshold"].iat[0])]):
            cm = confusion_matrix(g["y_true"], (g["p_intencao"] >= threshold).astype(int), labels=[1, 0])
            pct = cm / cm.sum(axis=1, keepdims=True)  # % de cada linha (classe verdadeira)
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=11,
                            color="white" if pct[i, j] > 0.5 else "black",
                            fontweight="bold" if i == j else "normal")
            ax.set_xticks([0, 1], ["intenção", "repouso"])
            ax.set_yticks([0, 1], ["intenção", "repouso"])
            ax.set_xlabel("resposta do modelo")
            ax.set_ylabel("verdade")
            ax.set_title(f"{names[model]}, corte {label} ({threshold:.2f})", fontsize=10)
    fig.tight_layout()
    save(fig, "matriz_confusao.png")


def fig_roc(test):
    names = {"eegconformer": "EEGConformer", "mlp": "perceptron multicamadas (MLP)"}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for model in MODELS:
        g = test[test["model"] == model]
        fpr, tpr, _ = roc_curve(g["y_true"], g["p_intencao"])
        auc = roc_auc_score(g["y_true"], g["p_intencao"])
        line, = ax.plot(100 * fpr, 100 * tpr, label=f"{names[model]}, área = {auc:.2f}".replace(".", ","))
        sens = g.loc[g["y_true"] == 1, "y_pred"].mean()
        spec = 1 - g.loc[g["y_true"] == 0, "y_pred"].mean()
        ax.plot(100 * (1 - spec), 100 * sens, "o", ms=8, color=line.get_color())
    ax.plot([0, 100], [0, 100], ls="--", c="gray", label="chute, área = 0,50")
    ax.plot([], [], "o", c="black", label="limiar escolhido na validação")
    ax.set(xlabel="repousos confundidos com intenção (%)\n(1 − especificidade)",
           ylabel="intenções detectadas (%)\n(sensibilidade)",
           title="Curva ROC no teste (A08 e A09)", xlim=(0, 100), ylim=(0, 100), aspect="equal")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "curva_roc.png")


def fig_tsne(emb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for subject, g in emb.groupby("subject_id"):
        axes[0].scatter(g["tsne_x"], g["tsne_y"], s=2, alpha=0.5, label=subject)
    for window, g in emb.groupby("window"):
        axes[1].scatter(g["tsne_x"], g["tsne_y"], s=2, alpha=0.4, label=window)
    for ax, title in zip(axes, ["cor = sujeito", "cor = janela"]):
        ax.set_title(f"t-SNE dos embeddings do CBraMod ({title})")
        ax.set_xticks([]), ax.set_yticks([])
        ax.legend(markerscale=5, fontsize=8)
    fig.tight_layout()
    save(fig, "tsne.png")


def fig_vizinhos(top1):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for pool, label in [("treino", "1º vizinho no treino"), ("todos", "1º vizinho em todas")]:
        ax.hist(top1.loc[top1["pool"] == pool, "cosine_sim"], bins=60, alpha=0.6, label=label)
    ax.axvline(0.99, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("cosseno"), ax.set_ylabel("amostras de teste"), ax.legend()
    save(fig, "semelhanca_vizinhos.png")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    con = connect()
    pred = load_predictions(con)
    test = pred[pred["split"] == "teste"]
    modelos = section_modelos(pred)
    chosen = max(MODELS, key=lambda m: modelos[m]["auc_validacao"])  # escolha na validação
    embeddings, emb, top1 = section_embeddings(con, pred)
    metrics = {"dados": section_dados(con), "qualidade": section_qualidade(con),
               "divisao": section_pipeline(con), "modelos": modelos,
               "comparacoes": section_comparacoes(pred, chosen), "embeddings": embeddings}
    metrics = clean(metrics)
    (OUT_DIR / "metricas.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig_sinal()
    fig_classes(con)
    fig_treino(con)
    fig_confusao(test)
    fig_roc(test)
    fig_tsne(emb)
    fig_vizinhos(top1)
    print(f"{count_leaves(metrics)} métricas -> outputs/metricas.json; "
          f"{len(FIGURES)} figuras -> figuras/ ({', '.join(FIGURES)})")


if __name__ == "__main__":
    main()
