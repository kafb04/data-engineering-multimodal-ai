# EEG Motor Movement/Imagery Dataset — ficha técnica e EDA

Entrega da Aula 01 (Ambiente, Datasets Abertos e Princípios FAIR).

## 1. Nome e fonte

**EEG Motor Movement/Imagery Dataset (`eegmmidb`)**, versão 1.0.0 — PhysioNet.
Depositado por Gerwin Schalk, publicado em 9 de setembro de 2009, coletado com o sistema
BCI2000 (descrito em Schalk et al., 2004).

- URL: <https://physionet.org/content/eegmmidb/1.0.0/>
- DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)
- RRID: `SCR_007345`
- Versão: **1.0.0** 

## 2. Licença

**Open Data Commons Attribution License v1.0 (ODC-BY 1.0).** Permite uso, redistribuição e
obras derivadas, inclusive comerciais, exigindo apenas atribuição da fonte. Não exige
credenciamento nem login — o download é direto por HTTPS.

## 3. Variáveis principais

| Variável | Tipo | Formato / unidade | Faixa |
|---|---|---|---|
| Sinal EEG | contínuo | EDF+ 16 bits; `float64` em Volts no MNE | −376 a +595 µV (medido na amostra); σ ≈ 53 µV |
| Canais | categórico | 64 eletrodos, sistema 10-10 | `Fc5.`…`Iz.` no arquivo bruto |
| Frequência de amostragem | constante | Hz | 160 Hz |
| Anotações | categórico | canal de eventos EDF+ | `T0` (repouso), `T1`, `T2` |
| Duração por run | contínuo | segundos | ~61 s (baseline) / ~125 s (tarefa) |
| Duração por trial | contínuo | segundos | ~4,1 s |
| Sujeito | identificador | `S001`–`S109` | 109 valores |
| Run | identificador | `R01`–`R14` | 14 por sujeito |

Os nomes de canal vêm com sufixo de ponto (`Fc5.`) e exigem
`mne.datasets.eegbci.standardize()` antes de qualquer análise espacial.

Significado de `T1`/`T2` por run:

| Runs | Tarefa | `T1` | `T2` |
|---|---|---|---|
| R01, R02 | linha de base (olhos abertos / fechados) | — | — |
| R03, R07, R11 | execução motora real | mão esquerda | mão direita |
| R04, R08, R12 | imagética motora | mão esquerda | mão direita |
| R05, R09, R13 | execução motora real | ambas as mãos | ambos os pés |
| R06, R10, R14 | imagética motora | ambas as mãos | ambos os pés |

## 4. Tamanho

- 109 sujeitos × 14 runs = **1.526 arquivos EDF+**
- **~3,4 GB** descompactado (~1,9 GB comprimido), ~26 h de gravação
- ~1,3 MB por arquivo de baseline, ~2,6 MB por run de tarefa
- **~19,6 mil trials** (~15 por run de tarefa, ~180 por sujeito)

Amostra usada nesta EDA: 5 sujeitos × 5 runs = 25 arquivos (~56 MB), com 150 trials `T1`,
150 `T2` e 305 eventos `T0`.

## 5. Riscos de privacidade

Desidentificado, mas **não anonimizado de forma irreversível**.

Não há identificadores diretos nem dados demográficos — apenas o ID sequencial
`S001`–`S109`. O acesso é aberto sem credenciamento, o que indica baixo risco na avaliação
dos curadores.

O risco residual é o **EEG fingerprinting**: padrões espectrais individuais são estáveis o
bastante para reidentificar uma pessoa entre gravações. A reidentificação exigiria que o
atacante já tivesse uma gravação de EEG rotulada do mesmo indivíduo, então na prática o
risco acadêmico é baixo — mas o sinal bruto não deve ser tratado como dado anônimo em
pipelines que o combinem com outras fontes.

## 6. Uso clínico e ML

**Aplicação.** BCI para reabilitação motora pós-AVC (imagética realimentada dirigindo
neuroplasticidade) e comunicação assistiva em pacientes com deficiência motora severa
(ELA, lesão medular alta).

**Pergunta de pesquisa.** É possível decodificar a intenção de movimento a partir do EEG —
identificar qual tarefa a pessoa está imaginando, sem movimento executado? E quanto a
modulação da imagética é mais fraca que a da execução real?

**Entradas.** Épocas de `64 canais × tempo` (~4 s a 160 Hz ≈ 656 amostras por canal), com
foco nos canais motores C3/Cz/C4 e nas bandas mu (8–12 Hz) e beta (13–30 Hz).

**Alvo.** A classe da tarefa no trial: `T1` vs. `T2` (ex.: mão esquerda vs. direita), ou
tarefa vs. repouso (`T0`).

**Métodos já relatados.** CSP + LDA (baseline clássico de BCI), classificadores
Riemannianos (matrizes de covariância, tangent space), CNNs para EEG (EEGNet,
ShallowConvNet) e híbridos CNN+LSTM / Transformers. O dataset é um dos benchmarks padrão
do [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html).

## 7. Avaliação FAIR

| Princípio | Nota (1–5) | Justificativa |
|---|:---:|---|
| Findable | 5 | Dois identificadores persistentes: DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P) e RRID `SCR_007345`. URL versionada (`/content/eegmmidb/1.0.0/`), com metadados descritivos completos — título, autor, data de publicação, versão, licença e citação. Indexado no MOABB como [`PhysionetMI`](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html), portanto localizável por ferramenta e não só por busca textual. |
| Accessible | 5 | HTTPS aberto, sem login, credenciamento ou DUA — confirmado ao baixar os 25 arquivos da amostra sem cabeçalho de autenticação (seção 10). Os metadados e a listagem de arquivos permanecem acessíveis sem baixar os dados, atendendo A2. |
| Interoperable | 4 | EDF+ é formato aberto e padrão em eletrofisiologia, lido por MNE, EEGLAB e FieldTrip sem conversão. Perde 1 ponto por três lacunas **verificadas nos arquivos**: (a) não segue [BIDS-EEG](https://bids-specification.readthedocs.io/) — não há `dataset_description.json`, `participants.tsv` nem `*_events.tsv`; (b) os nomes de canal vêm como `Fc5.`, `C3..` e só viram `FC5`/`C3` após `eegbci.standardize` (ver `load_raw` no notebook); (c) **não há coordenadas de eletrodo** — `raw.info["dig"]` é `None` e a posição de C3 é `[nan, nan, nan]`, de modo que qualquer análise espacial depende de um template externo. |
| Reusable | 4 | Licença ODC-BY 1.0 declarada de forma explícita, proveniência documentada (BCI2000, Schalk et al., 2004) e protocolo experimental descrito. Perde 1 ponto por: (a) os 6 sujeitos com anotações inconsistentes **não estarem sinalizados nos metadados oficiais** — constam apenas na literatura secundária (Kim et al., 2024); (b) ausência de variáveis demográficas, inclusive **lateralidade**, que condiciona a interpretação de tarefas motoras; (c) **não haver changelog nem versão anterior** — só a 1.0.0, sem registro de correções. |

**Média: 4,5 / 5.** As perdas se concentram em *interoperability* e *reusability* e têm a
mesma raiz: o dataset é de 2004 e antecede o BIDS e a prática de publicar notas de
qualidade junto com os dados.

## 8. Limitações conhecidas

### Específicas deste conjunto

- **Sem variáveis demográficas.** Não há idade, sexo, etnia nem lateralidade dos 109
  voluntários. A lateralidade é a mais custosa: ela condiciona a assimetria esperada em
  tarefas motoras, e sem ela não há como controlar esse fator.
- **6 dos 109 sujeitos (5,5 %) têm anotações inconsistentes** — `S088`, `S089`, `S092`,
  `S100`, `S104`, `S106` (Kim et al., 2024) — e não estão sinalizados nos metadados
  oficiais. Ficam em `PROBLEMATIC_SUBJECTS` e são filtrados no notebook.
- **Datas reais de aquisição indisponíveis.** Os 25 arquivos da amostra, de 5 sujeitos
  diferentes, trazem todos `meas_date = 2009-08-12` — carimbo de conversão em lote. Não há
  como avaliar o período de coleta.
- **Sessão única por sujeito**, o que impede medir estabilidade entre sessões — o principal
  obstáculo prático de BCIs baseadas em imagética.
- **Runs de baseline com estrutura diferente** das de tarefa (~61 s e um evento contínuo,
  contra ~125 s e 30 eventos), gerando valores ausentes nas contagens por classe.
- **160 Hz de amostragem** limitam a análise a menos de 80 Hz (Nyquist). Suficiente para as
  bandas mu e beta, que sustentam a decodificação motora, mas inviabiliza gama alta.

### Inerentes ao gênero, não a este conjunto

Valem para praticamente todos os conjuntos abertos de imagética motora, e não pesam na
avaliação FAIR — mas condicionam o que se pode concluir:

- Apenas voluntários saudáveis, sem pacientes da população-alvo (pós-AVC, ELA, lesão
  medular). Desempenho aqui não demonstra generalização clínica.
- Trials curtos (~4,1 s) em ambiente laboratorial controlado, o que tende a superestimar a
  acurácia frente ao uso real, com ruído, movimento e fadiga.
- Anotações marcam apenas eventos, sem rótulo de qualidade nem marcação de artefato.

## 9. Ambiente reprodutível

Python 3.12.13, com as dependências fixadas em [`requirements.txt`](requirements.txt):
`mne`, `numpy`, `pandas`, `matplotlib` e `notebook`.

```powershell
uv venv venv --python 3.12
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## 10. Dados

A amostra está versionada em `data/raw/`: 25 arquivos EDF+ (~56 MB) dos sujeitos S001–S005,
runs R01, R03, R04, R08 e R12. 

## 11. Como executar a EDA

Com o ambiente ativado:

```powershell
jupyter notebook notebooks\01_eda_eegmmidb.ipynb
```

## 12. Estrutura

```
.
├── README.md                        # ficha técnica e instruções
├── requirements.txt                 # dependências fixadas
├── notebooks/01_eda_eegmmidb.ipynb  # EDA executável do zero
└── data/raw/S0XX/S0XXR0Y.edf        # amostra
```
