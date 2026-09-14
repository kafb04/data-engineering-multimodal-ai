# BCI Competition IV 2a — ficha técnica e EDA

Dataset da tese (**decodificação da intenção de movimento**): imagética motora de 4 classes
em sujeitos saudáveis, carregado em formato pronto para ML via
[MOABB](https://moabb.neurotechx.com/) / [braindecode](https://braindecode.org/).

> A entrega anterior baseada no eegmmidb (EDF) foi arquivada em
> [`archive/aula01_eegmmidb/`](archive/aula01_eegmmidb/). Este repositório passou a ser
> exclusivamente sobre o BCI IV 2a.

## 1. Nome e fonte

**BCI Competition IV — dataset 2a** ("Graz data set A"), identificador
[`001-2014`](http://bnci-horizon-2020.eu/database/data-sets) no BNCI Horizon 2020 e
[`BNCI2014_001`](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)
no MOABB. Coletado no Institute of Neural Engineering (Laboratory of Brain-Computer
Interfaces), TU Graz.

- Descrição original: Brunner, Leeb, Müller-Putz, Schlögl & Pfurtscheller (2008).
- Citação recomendada: Tangermann et al. (2012), *Review of the BCI Competition IV*,
  Frontiers in Neuroscience — DOI [10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055).
- Fonte original da competição: <http://www.bbci.de/competition/iv/>

## 2. Licença

**Creative Commons Attribution-NoDerivatives 4.0 (CC BY-ND 4.0)** (conforme BNCI Horizon /
MOABB). Permite uso e redistribuição **do dataset original** com atribuição, mas a cláusula
**ND (NoDerivatives)** proíbe distribuir versões modificadas/derivadas dos dados.

> **Implicação prática:** não se deve versionar neste repositório uma versão processada
> (filtrada, epocada, convertida) do 2a — isso seria um derivado. Por isso os dados **não
> ficam no Git**; são baixados pelo MOABB no ambiente de cada pessoa (ver seção 10). Treinar
> modelos e publicar resultados/figuras derivados da análise não é afetado.

## 3. Variáveis principais

| Variável | Tipo | Formato / unidade | Faixa |
|---|---|---|---|
| Sinal EEG | contínuo | GDF (BioSig); `float64` em Volts no MNE | ±100 µV (sensibilidade do amplificador) |
| Canais EEG | categórico | 22 eletrodos (subconjunto 10-20) | `Fz`, `C3`, `Cz`, `C4`, … |
| Canais EOG | categórico | 3 eletrodos monopolares (controle de artefato) | `EOG-left/central/right` |
| Frequência de amostragem | constante | Hz | 250 Hz |
| Filtragem (aquisição) | — | passa-banda + notch | 0,5–100 Hz; notch 50 Hz |
| Classes | categórico | rótulo do trial | mão esquerda, mão direita, pés, língua |
| Sessão | identificador | treino (`T`) / avaliação (`E`) | 2 por sujeito, dias diferentes |
| Sujeito | identificador | `A01`–`A09` | 9 valores |

Estrutura de cada sessão: **6 runs × 48 trials (12 por classe) = 288 trials**.

Paradigma temporal do trial: cruz de fixação + bipe em `t=0`; seta de indicação (cue) em
`t=2 s` por 1,25 s; **imagética motora de `t≈2 s` até `t=6 s`** (~4 s de janela); pausa curta.

## 4. Tamanho

- 9 sujeitos × 2 sessões × 288 trials = **5.184 trials** rotulados.
- 4 classes balanceadas por desenho (72 trials por classe por sessão).
- Formato **GDF** (também disponível em `.mat` no BNCI). Os rótulos da sessão de avaliação,
  retidos durante a competição, hoje são públicos.

Amostra usada na EDA: 3 sujeitos (ver [`notebooks/02_eda_bci_iv_2a.ipynb`](notebooks/02_eda_bci_iv_2a.ipynb)).

## 5. Riscos de privacidade

Desidentificado (sujeitos `A01`–`A09`, sem demografia publicada). Não há identificadores
diretos. O risco residual é o **EEG fingerprinting** — padrões espectrais individuais são
estáveis o bastante para reidentificação entre gravações —, mas exigiria que o atacante já
tivesse um EEG rotulado do mesmo indivíduo, então o risco acadêmico é baixo. Ainda assim, o
sinal bruto não deve ser tratado como anônimo em pipelines que o combinem com outras fontes.

## 6. Uso clínico e ML

**Aplicação.** BCI de imagética motora para reabilitação e comunicação assistiva em pessoas
com deficiência motora severa (AVC, ELA, lesão medular). É o benchmark mais usado da área.

**Pergunta de pesquisa.** É possível decodificar a intenção de movimento a partir do EEG —
qual dos 4 movimentos a pessoa está imaginando — sem movimento executado?

**Entradas.** Épocas de `22 canais × tempo` (~4 s a 250 Hz ≈ 1.000 amostras por canal), com
foco nos canais motores C3/Cz/C4 e nas bandas mu (8–12 Hz) e beta (13–30 Hz).

**Alvo.** A classe do trial: mão esquerda, mão direita, pés ou língua (4 classes).

**Métodos já relatados.** CSP + LDA (baseline clássico), classificadores Riemannianos
(covariância / tangent space) e CNNs para EEG (EEGNet, ShallowConvNet, Deep4Net) — todos
disponíveis no [braindecode](https://braindecode.org/). Protocolo padrão: **within-subject**
(treina na sessão `T`, testa na `E`); estudos modernos também exploram **cross-subject**.

## 7. Avaliação FAIR

| Princípio | Nota (1–5) | Justificativa |
|---|:---:|---|
| Findable | 5 | DOI [10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055), id persistente `001-2014` no BNCI Horizon e indexação no MOABB como [`BNCI2014_001`](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) — localizável por ferramenta, não só por busca textual. |
| Accessible | 5 | Download aberto por HTTPS no BNCI/TU Graz, sem login nem DUA, e carga em uma linha via MOABB. Metadados descritos na página do dataset. |
| Interoperable | 4 | GDF é formato aberto de eletrofisiologia (BioSig), lido por MNE/MOABB sem conversão. Perde 1 ponto por não seguir [BIDS-EEG](https://bids-specification.readthedocs.io/) e por não trazer coordenadas de eletrodo padronizadas (a montagem depende de template externo). |
| Reusable | 3 | Proveniência bem documentada (TU Graz, protocolo detalhado) e licença explícita, **mas a cláusula ND da CC BY-ND 4.0 restringe a redistribuição de versões derivadas** dos dados — limitação real de reúso. Some-se a ausência de variáveis demográficas (idade, sexo, lateralidade). |

**Média: 4,25 / 5.** As perdas se concentram em *interoperability* (pré-BIDS) e
*reusability* (licença ND e falta de demografia).

## 8. Limitações conhecidas

### Específicas deste conjunto

- **Apenas 9 sujeitos.** Poucos sujeitos limitam estudos de generalização cross-subject e
  pré-treino — em compensação, há ~576 trials por sujeito, favorecendo o protocolo
  within-subject.
- **Licença CC BY-ND 4.0.** A cláusula NoDerivatives impede publicar/redistribuir uma versão
  processada dos dados (por isso o repo não versiona os dados).
- **Artefatos de EOG.** Há 3 canais de EOG justamente porque o piscar/movimento ocular
  contamina o EEG frontal; a remoção de artefato é parte esperada do pipeline.
- **Sem variáveis demográficas** (idade, sexo, lateralidade) — a lateralidade condiciona a
  assimetria esperada em tarefas motoras e não pode ser controlada.
- **Apenas 2 sessões** por sujeito (treino/avaliação), o que limita a análise de estabilidade
  entre múltiplos dias.

### Inerentes ao gênero, não a este conjunto

- Apenas voluntários saudáveis, sem pacientes da população-alvo (AVC, ELA, lesão medular).
  Desempenho aqui não demonstra generalização clínica.
- Trials curtos (~4 s) em ambiente laboratorial controlado, o que tende a superestimar a
  acurácia frente ao uso real, com ruído, movimento e fadiga.

## 9. Ambiente reprodutível

Python 3.12.13, com as dependências em [`requirements.txt`](requirements.txt): `mne`, `numpy`,
`pandas`, `matplotlib`, `notebook` e — para o 2a — `moabb`, `braindecode` e `torch`.

```powershell
uv venv venv --python 3.12
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## 10. Dados

Os dados **não são versionados** neste repositório (a licença CC BY-ND desaconselha
redistribuir derivados, e o volume é grande). O MOABB baixa o dataset para o cache local
`~/mne_data` na primeira execução dos notebooks — requer internet:

```python
from braindecode.datasets import MOABBDataset
MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])  # baixa e carrega
```

## 11. Como executar

Com o ambiente ativado:

```powershell
jupyter notebook notebooks\02_eda_bci_iv_2a.ipynb   # EDA (amostra de 3 sujeitos)
jupyter notebook notebooks\03_load_bci_iv_2a.ipynb  # carga completa -> DataLoader PyTorch
```

## 12. Estrutura

```
.
├── README.md                            # esta ficha técnica
├── requirements.txt                     # dependências fixadas
├── notebooks/
│   ├── 02_eda_bci_iv_2a.ipynb           # EDA do 2a
│   └── 03_load_bci_iv_2a.ipynb          # download -> preprocess -> DataLoader
└── archive/
    ├── README.md                        # nota sobre o material arquivado
    └── aula01_eegmmidb/                  # entrega da Aula 01 (eegmmidb), preservada
        ├── README.md                    # datasheet FAIR do eegmmidb
        ├── notebooks/01_eda_eegmmidb.ipynb
        └── data/raw/S0XX/S0XXR0Y.edf     # amostra EDF
```
