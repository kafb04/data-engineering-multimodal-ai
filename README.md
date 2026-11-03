# Decodificação da intenção de movimento — BCI Competition IV 2a

Sinal cerebral (eletroencefalograma) de 9 pessoas imaginando movimentos. Objetivo: detectar, só
pelo sinal, que a pessoa **quer** se mexer.

```mermaid
flowchart LR
    T1["Tarefa 1<br/>conhecer os dados"] --> T2["Tarefa 2<br/>organizar em tabelas"]
    T2 --> T3["Tarefa 3<br/>checar a qualidade"]
    T3 --> T4["Tarefa 4<br/>redes neurais e avaliação"]
    T4 --> T5["Tarefa 5<br/>embeddings"]
```

## Como rodar

```powershell
uv sync --locked                                          # instala as bibliotecas
uv run python -m src.ingest                               # Tarefa 2
uv run dbt build --project-dir dbt --profiles-dir dbt     # Tarefa 3
uv run python -m src.train                                # Tarefa 4
uv run python -m src.embed                                # Tarefa 5
uv run jupyter notebook notebooks/                        # gráficos e análises
```

---

## Tarefa 1 — Conhecer os dados

| item | resumo |
|---|---|
| Fonte | Universidade de Tecnologia de Graz — [artigo](https://doi.org/10.3389/fnins.2012.00055) |
| Gravação | 22 eletrodos na cabeça + 3 em volta dos olhos, 250 medições por segundo |
| Movimentos imaginados | mão esquerda, mão direita, pés, língua |
| Tamanho | 9 pessoas × 2 dias × 288 tentativas = **5.184 tentativas** |
| Licença | proíbe redistribuir versões modificadas, por isso o sinal não fica no repositório |

**Dados abertos (nota de 1 a 5):** fácil de encontrar 5 · fácil de acessar 5 · fácil de combinar 4 ·
fácil de reutilizar 3 → **média 4,25**.

Sinal dos eletrodos C3, Cz e C4 (lado esquerdo, centro e lado direito da cabeça) em duas tentativas,
e as 4 classes, equilibradas:

<img src="figuras/sinal_c3_cz_c4.png" width="420" alt="Sinal dos eletrodos C3, Cz e C4 em duas tentativas">
<img src="figuras/distribuicao_classes.png" width="400" alt="Quantidade de tentativas por classe">

**Limitações:** só 9 pessoas saudáveis, 2 dias de gravação, piscadas atrapalham o sinal.

---

## Tarefa 2 — Organizar em tabelas

As informações de cada tentativa (sem o sinal) ficam em uma tabela central ligada a 4 tabelas de descrição.

```mermaid
erDiagram
    pessoa    ||--o{ tentativa : ""
    dia       ||--o{ tentativa : ""
    bloco     ||--o{ tentativa : ""
    movimento ||--o{ tentativa : ""

    tentativa {
        texto   pessoa
        texto   dia
        texto   bloco
        numero  movimento
        decimal inicio_em_segundos
        sim_nao tem_ruido
    }
    pessoa {
        numero idade
        texto  sexo
        texto  mao_dominante "vazio"
    }
    movimento {
        texto nome
        texto parte_do_corpo
    }
    dia {
        texto papel "treino ou avaliação"
    }
    bloco {
        numero numero_do_bloco
    }
```

O formato Parquet ficou **6 vezes menor** que a planilha comum e as consultas, **20 vezes mais rápidas**.

---

## Tarefa 3 — Checar a qualidade

```mermaid
flowchart LR
    B["Bronze<br/>arquivos originais"] --> P["Prata<br/>tabelas conferidas"]
    P --> O["Ouro<br/>tentativas sem ruído"]
    P -.->|32 testes automáticos| OK(["todos passam"])
```

| verificação | resultado |
|---|---|
| Linhas repetidas | nenhuma |
| Valores inválidos (18 regras) | nenhum |
| Tentativas com ruído | 488 de 5.184 (**9,4%**), marcadas e deixadas de fora |

---

## Tarefa 4 — Redes neurais e avaliação

**Pergunta:** o modelo diz se a pessoa está em **repouso** ou com **intenção** de movimento, em
pessoas que ele nunca viu? Cada tentativa dá 2 trechos de 1,5 segundo do sinal: um logo antes da seta
na tela (repouso) e um de 1,5 a 3 segundos depois dela (intenção).

### Escolhas

```mermaid
flowchart LR
    E2["treino supervisionado"]:::sim --> E4["inferência"]:::sim
    classDef sim fill:#2E7D32,color:#fff,stroke:#1B5E20
```

| tema | **escolhi** | por quê |
|---|---|---|
| Etapas de treino | **treino supervisionado + inferência** | os modelos são treinados do zero, direto com os rótulos de repouso e intenção, e depois usados para prever |
| Rede neural | **EEGConformer** × **perceptron multicamadas (MLP)** | o primeiro lê o sinal inteiro (convolução + atenção); o segundo, só a força dos ritmos cerebrais. Mostra se ler o sinal completo ajuda |
| Pré-processamento | **z-score**, só com o treino | deixa cada variável com média 0 e desvio 1; usar só o treino evita vazar informação do teste |
| Contra overfitting | **dropout, penalização dos pesos e parada antecipada** | desliga neurônios ao acaso, impede pesos grandes e para quando a validação não melhora: evita que o modelo decore o treino |
| Divisão | **por pessoa**: treino A01–A05, validação A06–A07, teste A08–A09 | testar em quem o modelo nunca viu |
| Testes estatísticos | **McNemar, DeLong e qui-quadrado, com correção de Holm** | McNemar compara os acertos dos dois modelos; DeLong, as áreas sob a curva ROC; qui-quadrado, o acerto entre grupos; Holm corrige por fazer vários testes |

### Resultados

**Overfitting:** a perda no treino cai sem parar, mas a da validação não acompanha. O modelo
guardado é o da época com maior área sob a curva ROC na validação (linha tracejada).

<img src="figuras/curvas_treino.png" width="720" alt="Perda e área sob a curva ROC por época">

**Curva ROC no teste:** mostra sensibilidade × erros no repouso em todos os limiares. Quanto mais
acima da linha tracejada (acaso), melhor. O ponto marca o limiar escolhido na validação.

<img src="figuras/curva_roc.png" width="400" alt="Curva ROC do EEGConformer e do perceptron multicamadas no teste">

**Área sob a curva ROC:** 0,5 = chute, 1 = acerta sempre.

```mermaid
---
config:
  xyChart:
    width: 720
    height: 260
  themeVariables:
    xyChart:
      plotColorPalette: "#4C78A8, #E45756"
---
xychart-beta
    title "Área sob a curva ROC (linha vermelha = chute)"
    x-axis ["EEGConformer validação", "EEGConformer teste", "Perceptron validação", "Perceptron teste"]
    y-axis 0.4 --> 0.8
    bar [0.761, 0.600, 0.599, 0.561]
    line [0.5, 0.5, 0.5, 0.5]
```

| no teste | sensibilidade | especificidade | acurácia |
|---|---|---|---|
| **EEGConformer** | 78,0% | 37,2% | **57,6%** |
| Perceptron multicamadas | 74,3% | 35,8% | 55,0% |

Os dois acertam a maioria das intenções e erram a maioria dos repousos: tendem a responder "intenção".

**Matriz de confusão no teste** (linha = verdade, coluna = resposta do modelo). Com o limiar da
validação, os dois modelos respondem "intenção" na maioria dos repousos. O limiar que servia para as
pessoas da validação não serve para as do teste:

<img src="figuras/matriz_confusao.png" width="520" alt="Matriz de confusão dos dois modelos no teste">

- O EEGConformer separa melhor as classes (DeLong, p = 0,018), mas os dois ficam pouco acima do chute.
- A nota cai de 0,76 na validação para 0,60 no teste: **cada cérebro é diferente** (domain shift).
- O acerto muda entre as pessoas de teste (61% × 54%, p = 0,002), mas não com o movimento nem com o ruído: o que pesa é a pessoa.

---

## Tarefa 5 — Embeddings, CLIP e modelos de visão e linguagem

**Ideia:** um modelo já pré-treinado transforma cada trecho em um **embedding** (4.400 números que
o resumem). Trechos com embeddings parecidos são **vizinhos**.

### Escolhas

```mermaid
flowchart LR
    Q(["sinal cerebral"]) --> O1["embeddings +<br/>busca de vizinhos"]:::sim
    classDef sim fill:#2E7D32,color:#fff,stroke:#1B5E20
```

| tema | **escolhi** | por quê |
|---|---|---|
| Abordagem | **embeddings + busca de vizinhos** | funciona direto no sinal elétrico do cérebro |
| Modelo | **CBraMod** | pré-treinado sem os dados deste projeto (não viu o teste), pesos públicos |
| Uso do modelo | **congelado** | os pesos não mudam: testa só o que ele já sabe do pré-treino |
| Busca | **exata** | compara todos os trechos entre si; 10 mil trechos cabem em segundos |
| Quase duplicatas | **os 4 passos**: vizinho, ordenar, inspecionar, corrigir | procurar cópias entre treino e teste |

### Resultados

| verificação | resultado |
|---|---|
| Cópias entre treino e teste | **nenhuma** |
| Vizinhos da mesma pessoa (acaso: 11%, 1 em 9 pessoas) | **25,4%** |
| Vizinhos do mesmo tipo de trecho (acaso: 50%) | 52,3% |

Semelhança de cada trecho de teste com o vizinho mais parecido: nenhum chega perto de 0,99
(linha tracejada), o limite para ser considerado cópia.

<img src="figuras/semelhanca_vizinhos.png" width="520" alt="Histograma da semelhança com o vizinho mais parecido">

O modelo **agrupa por pessoa**, não por intenção:

```mermaid
---
config:
  xyChart:
    width: 560
    height: 260
  themeVariables:
    xyChart:
      plotColorPalette: "#4C78A8, #E45756"
---
xychart-beta
    title "Vizinhos da mesma pessoa, em % (linha vermelha = acaso)"
    x-axis [A01, A02, A03, A04, A05, A06, A07, A08, A09]
    y-axis "%" 0 --> 50
    bar [14.5, 27.4, 34.8, 13.5, 26.0, 19.2, 12.7, 32.1, 48.1]
    line [11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1]
```

O mapa t-SNE (embeddings projetados em 2 dimensões) mostra o mesmo: colorindo por pessoa, A09
(amarelo) forma um grupo próprio; colorindo por tipo de trecho, repouso e intenção se misturam.

<img src="figuras/tsne.png" width="720" alt="Mapa t-SNE colorido por pessoa e por tipo de trecho">

Para separar repouso de intenção, cada trecho de teste recebe a classe da maioria dos seus 10
vizinhos. Essa votação fica abaixo dos dois modelos da Tarefa 4:

```mermaid
---
config:
  xyChart:
    width: 560
    height: 260
  themeVariables:
    xyChart:
      plotColorPalette: "#4C78A8, #E45756"
---
xychart-beta
    title "Área sob a curva ROC no teste (linha vermelha = chute)"
    x-axis ["EEGConformer", "Perceptron", "Vizinhos"]
    y-axis 0.4 --> 0.8
    bar [0.600, 0.561, 0.535]
    line [0.5, 0.5, 0.5]
```

Azul: trecho de teste. Laranja: o vizinho mais parecido no treino. Na primeira fileira, um repouso
tem como vizinho uma intenção: parecido para o modelo não quer dizer mesma classe.

<img src="data/embeddings/pares_vizinhos.png" width="700" alt="Trecho de teste ao lado do vizinho mais parecido no treino">

**Conclusões:**

- Não há cópias entre treino e teste, então os resultados da Tarefa 4 não foram inflados.
- O modelo reconhece **de quem é o sinal** (impressão digital cerebral), o que explica a queda da Tarefa 4.
- O modelo não reconhece a intenção: foi pré-treinado sem saber o que a pessoa fazia e usado congelado.
