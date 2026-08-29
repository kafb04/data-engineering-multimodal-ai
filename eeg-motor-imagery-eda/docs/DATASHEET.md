# Ficha Técnica: EEG Motor Movement/Imagery Dataset

## 1. Nome e fonte

**EEG Motor Movement/Imagery Dataset (eegmmidb)**, versão 1.0.0 — PhysioNet.
URL: <https://physionet.org/content/eegmmidb/1.0.0/>
DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)

Coletado por Schalk et al. com o sistema BCI2000, disponibilizado via
PhysioNet (Goldberger et al., 2000).

## 2. Licença

Open Data Commons Attribution License v1.0 (ODC-BY 1.0): permite uso,
cópia, redistribuição e obras derivadas, inclusive comercial, exigindo
apenas atribuição da fonte. Diferente de bases restritas do PhysioNet
(ex.: MIMIC-III), não exige credenciamento para download.

## 3. Variáveis principais

| Variável | Descrição |
|---|---|
| Sinal EEG | 64 canais, contínuo, em µV |
| Canais | Sistema 10-10, nomes exigem padronização (`mne.datasets.eegbci.standardize`) |
| Frequência de amostragem | 160 Hz |
| Anotações | `T0` (repouso), `T1`, `T2` (classe depende da corrida) |
| Sujeitos / corridas | S001–S109, R01–R14 por sujeito |
| Formato | EDF+ (sinal) + `.edf.event` (anotações) |

Corridas: R01–R02 baseline (olhos abertos/fechados); R03/R07/R11 execução
motora mão esquerda vs. direita; R04/R08/R12 imagética motora mão esquerda
vs. direita; R05/R09/R13 e R06/R10/R14 equivalentes para ambas as mãos vs.
ambos os pés. Trials de ~4,1 s.

## 4. Tamanho

109 sujeitos × 14 runs = 1.526 arquivos EDF+. ~3,4 GB descompactado
(1,9 GB comprimido), ~26h de gravação no total.

## 5. Riscos de privacidade

Acesso aberto, sem credenciamento — indício de baixo risco avaliado pelos
curadores. Não há dados demográficos publicados, só o ID `S001`–`S106`.
Risco residual: há literatura sobre "EEG fingerprinting" (reidentificação
por padrões espectrais individuais), então o sinal não é 100% anônimo se
cruzado com outra gravação identificada da mesma pessoa. Classificação:
desidentificado, mas não anonimizado de forma irreversível.

## 6. Uso clínico e ML

**Aplicação:** interfaces cérebro-computador (BCI) para reabilitação
motora pós-AVC e comunicação assistiva em pacientes com deficiência
motora severa.
**Pergunta de pesquisa:** dá para decodificar, a partir do EEG, qual
tarefa motora (real ou imaginada) a pessoa está executando?
**Entradas:** épocas do sinal (64 canais × tempo) ao redor de cada
evento, geralmente focadas nos canais motores C3/Cz/C4.
**Alvo:** classe da tarefa no trial (ex. mão esquerda vs. direita, `T1`
vs `T2`; ou tarefa vs. repouso).
**Métodos já usados na literatura:** CSP + LDA (baseline clássico de
BCI), classificadores Riemannianos, CNNs (EEGNet e variantes), e modelos
híbridos deep learning (CNN+LSTM, Transformers). O dataset é um dos
benchmarks padrão do framework [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html).

## 7. Avaliação FAIR

| Princípio | Nota (1–5) | Justificativa |
|---|:---:|---|
| Findable | 5 | DOI persistente, indexado no PhysioNet, referenciado por dezenas de artigos e pelo MOABB |
| Accessible | 5 | Download HTTPS aberto, sem login/credenciamento, metadados sempre acessíveis |
| Interoperable | 4 | Formato EDF+ padrão (MNE/EEGLAB), mas não segue BIDS-EEG e nomes de canal exigem padronização |
| Reusable | 4 | Licença aberta e clara, proveniência documentada; falta documentação oficial dos problemas de qualidade conhecidos (ver limitações) |

**Média: 4,5 / 5**

## 8. Limitações conhecidas

- Amostra pequena (109 voluntários saudáveis, sem dados demográficos) — pouco representativa de pacientes reais (pós-AVC etc.).
- 6 sujeitos com inconsistências conhecidas (S088, S089, S092, S100, S104, S106), identificados por Kim et al. (2024) e geralmente excluídos.
- Dados de 2004, protocolo e eletrodos defasados frente a headsets atuais.
- Trials curtos e em ambiente controlado, o que tende a inflar a acurácia frente a uso real.
- Anotações só marcam eventos (T0/T1/T2), sem rótulo de qualidade/artefato.
