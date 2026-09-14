# Arquivo

Material da **Aula 01** (Ambiente, Datasets Abertos e Princípios FAIR), baseado no dataset
**EEG Motor Movement/Imagery (eegmmidb)** em formato EDF.

Foi arquivado quando a tese migrou para o **BCI Competition IV 2a** (ver
[`../README.md`](../README.md)), que é carregado via MOABB/braindecode em formato pronto para
ML. Nada aqui é usado pelo fluxo atual; fica preservado como histórico da entrega.

```
aula01_eegmmidb/
├── README.md                     # datasheet FAIR do eegmmidb
├── notebooks/01_eda_eegmmidb.ipynb
└── data/raw/S0XX/S0XXR0Y.edf     # amostra EDF (25 arquivos)
```

A mini-estrutura é autocontida: o notebook usa caminhos relativos (`../README.md`,
`../data/raw`) e roda a partir de `aula01_eegmmidb/notebooks/`.
