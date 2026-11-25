"""Testes do código que não depende do sinal: divisão, estatística, ingestão e orquestrador.

Uso: `uv run python -m unittest -v`.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import run_pipeline
from src.evaluate import holm, to_json
from src.ingest import build_fact_table
from src.samples import SEED, SUBJECTS, folds


class TestDivisao(unittest.TestCase):
    def test_cada_pessoa_e_testada_uma_vez(self):
        self.assertEqual(sorted(test for test, _, _ in folds()), SUBJECTS)

    def test_conjuntos_nao_se_sobrepoem(self):
        for test, val, train in folds():
            self.assertNotEqual(test, val)
            self.assertNotIn(test, train)
            self.assertNotIn(val, train)
            self.assertEqual(len(train), 7)


class TestEstatistica(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

    def test_holm_nao_passa_de_1(self):
        self.assertTrue((holm([0.6, 0.7, 0.9]) <= 1).all())

    def test_to_json(self):
        convertido = to_json({"a": np.float32(0.123456), "b": np.int64(3),
                              "c": [np.nan, np.bool_(True)]})
        self.assertEqual(json.loads(json.dumps(convertido)),
                         {"a": 0.1235, "b": 3, "c": [None, True]})


class TestIngestao(unittest.TestCase):
    def janela(self, run, starts):
        starts = np.array(starts)
        metadata = pd.DataFrame({"i_start_in_trial": starts, "i_stop_in_trial": starts + 1000,
                                 "i_trial_in_dataset": range(len(starts)), "target": 0})
        return SimpleNamespace(description={"subject": 4, "session": "0train", "run": run},
                               metadata=metadata, raw=SimpleNamespace(info={"sfreq": 250.0}))

    def test_fato_e_marcacao_de_ruido(self):
        windows = SimpleNamespace(datasets=[self.janela("1", [750, 2750]), self.janela("2", [500])])
        fact = build_fact_table(windows, {("0train", "1"): np.array([2750])})
        self.assertEqual(fact["trial_id"].tolist(),
                         ["A04_0train_run_1_t00", "A04_0train_run_1_t01", "A04_0train_run_2_t00"])
        self.assertEqual(fact["has_artifact"].tolist(), [False, True, False])
        self.assertEqual(fact["onset_s"].tolist(), [3.0, 11.0, 2.0])
        self.assertTrue((fact["duration_s"] == 4.0).all())


class TestOrquestrador(unittest.TestCase):
    def rodar(self, etapas, selecionadas=()):
        with (
            tempfile.TemporaryDirectory() as pasta,
            mock.patch.object(run_pipeline, "ETAPAS", etapas),
            mock.patch.object(run_pipeline, "REGISTROS", Path(pasta)),
            mock.patch("sys.stdout"),
        ):
            status = run_pipeline.rodar(selecionadas)
            registros = {arquivo.stem: json.loads(arquivo.read_text(encoding="utf-8"))
                         for arquivo in Path(pasta).glob("*.json")}
        return status, registros

    @staticmethod
    def etapa(codigo, depende_de=()):
        return {"comando": ["-c", codigo], "depende_de": list(depende_de), "entradas": [], "saidas": []}

    def test_falha_bloqueia_so_as_dependentes(self):
        etapas = {"a": self.etapa("pass"), "b": self.etapa("import sys; sys.exit(1)", ["a"]),
                  "c": self.etapa("pass", ["b"]), "d": self.etapa("pass", ["a"])}
        status, registros = self.rodar(etapas)
        self.assertEqual(status, {"a": "ok", "b": "falhou", "c": "nao_rodou", "d": "ok"})
        self.assertEqual(registros["c"]["motivo"], "depende de ['b']")

    def test_saida_ausente_conta_como_falha(self):
        ausente = Path(tempfile.gettempdir()) / "saida_que_a_etapa_nao_gerou.parquet"
        status, _ = self.rodar({"a": {**self.etapa("pass"), "saidas": [str(ausente)]}})
        self.assertEqual(status, {"a": "falhou"})

    def test_saida_antiga_conta_como_falha(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as arquivo:
            antiga = Path(arquivo.name)
        os.utime(antiga, (0, 0))
        try:
            status, _ = self.rodar({"a": {**self.etapa("pass"), "saidas": [str(antiga)]}})
        finally:
            antiga.unlink()
        self.assertEqual(status, {"a": "falhou"})

    def test_roda_so_as_etapas_pedidas(self):
        etapas = {"a": self.etapa("pass"), "b": self.etapa("pass", ["a"])}
        self.assertEqual(self.rodar(etapas, ["b"])[0], {"b": "ok"})

    def test_etapa_desconhecida(self):
        with self.assertRaises(ValueError):
            self.rodar({"a": self.etapa("pass")}, ["x"])

    def test_dag_do_projeto_e_valido(self):
        grafo = {nome: etapa["depende_de"] for nome, etapa in run_pipeline.ETAPAS.items()}
        for dependencias in grafo.values():
            self.assertLessEqual(set(dependencias), set(grafo))
        self.assertCountEqual(run_pipeline.TopologicalSorter(grafo).static_order(), grafo)

    def test_semente_igual_a_do_treino(self):
        self.assertEqual(run_pipeline.SEMENTE, SEED)


if __name__ == "__main__":
    sys.exit(unittest.main())
