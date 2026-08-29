# EDA — EEG Motor Movement/Imagery Dataset (PhysioNet)

Entrega da disciplina de Tratamento de Dados / Engenharia de Dados e Machine
Learning: escolha de um dataset de EEG, ficha técnica com avaliação FAIR,
ambiente reprodutível e notebook de exploração (EDA).

## Dataset escolhido

**EEG Motor Movement/Imagery Dataset (eegmmidb)** — PhysioNet, versão 1.0.0
DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)
URL: <https://physionet.org/content/eegmmidb/1.0.0/>

A ficha técnica completa (nome/fonte, licença, variáveis, tamanho, riscos de
privacidade, uso clínico/ML e avaliação FAIR com justificativa) está em
[`docs/DATASHEET.md`](docs/DATASHEET.md) (também disponível em
[`docs/DATASHEET.pdf`](docs/DATASHEET.pdf)).

## Estrutura do repositório

```
.
├── README.md                      # este arquivo
├── requirements.txt                # dependências fixadas (pip freeze)
├── docs/
│   ├── DATASHEET.md                # ficha técnica completa (fonte)
│   └── DATASHEET.pdf               # mesma ficha técnica, em PDF
├── src/
│   └── data_loading.py             # funções de download/carga/padronização (MNE)
├── notebooks/
│   └── 01_eda_eegmmidb.ipynb       # notebook de EDA, executável do zero
└── data/
    ├── raw/                        # cache local dos .edf baixados (gerado na 1ª execução)
    └── eda_summary.csv             # resumo gerado pelo notebook (gerado na execução)
```

## Ambiente reprodutível

### Opção A — local (venv)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_eegmmidb.ipynb
```

### Opção B — local (conda)

```bash
conda create -n eeg-eda python=3.11 -y
conda activate eeg-eda
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_eegmmidb.ipynb
```

### Opção C — nuvem (Google Colab / Kaggle Kernels)

1. Faça upload da pasta do projeto (ou clone o repositório) no ambiente.
2. Na primeira célula do notebook, instale as dependências:
   ```python
   !pip install -r requirements.txt
   ```
3. Execute as células normalmente — Colab e Kaggle Kernels têm acesso
   irrestrito à internet, então o download dos dados funciona sem configuração
   adicional.

## Sobre o download dos dados (importante)

O notebook baixa os dados **diretamente do PhysioNet** na primeira execução,
usando o mecanismo oficial `mne.datasets.eegbci.load_data` (acesso aberto, sem
necessidade de login/credenciamento). Os arquivos ficam em cache em
`data/raw/`; execuções seguintes reaproveitam o cache.

> **Nota de transparência:** este projeto foi montado em um ambiente de
> sandbox cuja política de rede bloqueia acesso direto a `physionet.org`
> (só libera PyPI, npm e GitHub). Por isso, toda a lógica de carregamento,
> limpeza e análise do notebook foi **validada ponta a ponta com dados
> sintéticos com a mesma estrutura do dataset real** (64 canais, 160 Hz,
> anotações `T0`/`T1`/`T2`, mesmas funções do `mne`) — sem nenhum erro de
> execução — mas o notebook em si **não foi pré-executado com os dados reais**.
> Ao rodar em um ambiente com internet normal (sua máquina, Colab ou Kaggle
> Kernels), a primeira execução baixa ~15–20 MB por sujeito/run selecionado
> (a amostra padrão do notebook usa 5 sujeitos × 3 runs) e deve levar poucos
> minutos.

Se preferir baixar manualmente para conferir antes de rodar o notebook:

```bash
# Exemplo: sujeito 1, run 4 (imagética motora, mão esquerda vs. direita)
curl -O https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf
```

## Como executar o notebook do zero

1. Configure o ambiente (Opção A, B ou C acima).
2. Abra `notebooks/01_eda_eegmmidb.ipynb`.
3. Execute todas as células em ordem (*Run All*). A primeira execução baixa a
   amostra de dados definida na Seção 2 do notebook (`SUBJECTS`, `RUNS` —
   ajustável).
4. O notebook gera:
   - tabela-resumo de cobertura/diversidade da amostra;
   - gráficos de balanceamento de classes;
   - visualização do sinal bruto e da montagem de eletrodos;
   - comparação espectral (PSD) entre classes de tarefa;
   - checagem de consistência (frequência de amostragem, nº de canais, NaNs);
   - `data/eda_summary.csv`, pronto para ser consumido em uma etapa posterior
     de modelagem de ML.

## Dependências principais

Ver `requirements.txt` para a lista completa e fixada (`pip freeze`). As
principais bibliotecas são:

- `mne` — leitura, padronização e análise de sinais EEG (formato EDF+)
- `numpy`, `pandas` — manipulação de dados
- `matplotlib`, `seaborn` — visualização
- `scikit-learn` — pronta para a etapa de modelagem (não usada nesta EDA)
- `jupyter` — execução do notebook

## Licença dos dados

O dataset é distribuído sob **Open Data Commons Attribution License v1.0
(ODC-BY 1.0)** — uso livre, inclusive comercial, mediante atribuição da fonte
original. Ver detalhes em `docs/DATASHEET.md`, seção 2.
