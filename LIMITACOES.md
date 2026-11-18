# Relatório de limitações

Vale para todo o projeto ([README](README.md)). Os números vêm de
[`outputs/metricas.json`](outputs/metricas.json).

## 1. Vieses

| viés | de onde vem | efeito |
|---|---|---|
| **Amostra** | 9 voluntários saudáveis, sem mão dominante registrada | nada garante que o sinal de um paciente com AVC, ELA ou lesão medular se comporte igual |
| **Laboratório** | tentativas curtas, com hora marcada, pessoa parada e olhando a tela | no uso real a pessoa se mexe, se distrai e decide sozinha quando agir; o acerto tende a ser menor |
| **Definição de "repouso"** | o repouso é a espera antes da seta, com a cruz na tela | a pessoa pode já estar se preparando; o modelo pode aprender "antes × depois do aviso", não "repouso × intenção" |
| **Sinais que não são intenção** | piscadas, movimento dos olhos e expectativa da seta aparecem no EEG frontal | parte do acerto pode vir daí; os eletrodos dos olhos ficaram fora, mas o efeito não foi medido |
| **Marcação de ruído** | feita pelos autores do conjunto, não refeita aqui | os testes de qualidade olham as tabelas, não o sinal: um eletrodo ruim passaria |
| **Quarentena desigual** | A06 perde 24,7% das tentativas; A01, 3,8% | treino e validação ficam com menos dados de quem tem mais ruído, e as classes deixam de ser exatamente equilibradas |
| **Regra diferente no teste** | treino e validação sem ruído; teste com tudo | escolha proposital (uso real), mas o teste é mais difícil que a validação |

## 2. Limites de generalização

- **Só 2 pessoas no teste (A08 e A09).** O resultado descreve essas duas pessoas. O acerto já muda
  entre elas (61% × 54%).
- **O intervalo de confiança é otimista.** Cada trecho entra como independente, mas os 2.304 trechos
  de teste vêm de 2 pessoas. A incerteza entre pessoas não está no intervalo.
- **Uma única divisão.** Treino A01–A05, validação A06–A07, teste A08–A09, uma semente (42). Outra
  divisão ou outra semente daria outros números; essa variação não foi medida.
- **O corte da validação não se transfere.** O corte de 0,14, escolhido em A06–A07, faz o modelo
  responder "intenção" em 70% das vezes em A08–A09. A área sob a curva ROC não depende do corte.
- **A nota do modelo não é probabilidade.** Não foi calibrada: 0,8 não quer dizer 80% de chance.
- **Só 2 dias por pessoa.** Não dá para saber se o modelo continua valendo semanas depois.
- **Um equipamento, um laboratório.** 22 eletrodos, 250 Hz, mesma montagem. Outro equipamento não
  foi testado.
- **Embeddings congelados.** O CBraMod foi usado sem ajuste, com trechos de 1 s. O resultado mede
  o que o pré-treino já sabia, não o que o modelo consegue aprender.

## 3. O que não se pode concluir

| não se pode dizer que... | por quê |
|---|---|
| o modelo detecta intenção de movimento em uma pessoa nova | área de 0,60 em 2 pessoas, pouco acima do chute (0,50) e com queda forte desde a validação (0,76) |
| o método serve para pacientes ou para uso clínico | só há voluntários saudáveis, em laboratório, e o acerto está longe do necessário |
| o EEGConformer é melhor que o MLP em geral | a diferença (0,60 × 0,56) vale para esta divisão e esta semente; uma só execução de cada |
| o CBraMod não serve para esta tarefa | só foi testado congelado e com votação de vizinhos; com ajuste fino o resultado pode ser outro |
| não há vazamento nenhum | a busca descarta cópias de trechos entre treino e teste; não testa outros caminhos, como a ordem das tentativas |
| o que o modelo aprendeu é atividade motora | não foi feita análise de quais eletrodos ou ritmos pesam na decisão |
| o sinal identifica uma pessoa | 25,4% dos vizinhos são da mesma pessoa (acaso: 11%): é um indício de "impressão digital", não um teste de reidentificação |
| o tipo de movimento ou o ruído não importam | não houve diferença significativa, mas com 2 pessoas o teste tem pouca força para achar uma |

## 4. Reprodução

- **Mesmo ambiente, mesmo resultado.** Sementes fixas e CPU. Entre versões de biblioteca, sistemas
  ou processadores diferentes, os pesos treinados podem mudar nas últimas casas e alterar algum
  número. Por isso os modelos, as predições e o registro da execução ficam no repositório.
- **Dados fora do repositório.** A licença (CC BY-ND 4.0) impede redistribuir o sinal e os vetores de
  embedding. A reprodução depende de o MOABB e o Hugging Face continuarem servindo os arquivos.
