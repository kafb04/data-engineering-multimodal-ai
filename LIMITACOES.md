# Relatório de limitações

Vale para todo o projeto ([README](README.md)). Os números vêm de
[`outputs/metricas.json`](outputs/metricas.json).

## 1. Vieses

| viés | de onde vem | efeito |
|---|---|---|
| **Amostra** | 9 voluntários saudáveis, sem mão dominante registrada | nada garante que o sinal de um paciente com AVC, ELA ou lesão medular se comporte igual |
| **Laboratório** | tentativas curtas, com hora marcada, pessoa parada e olhando a tela | no uso real a pessoa se mexe, se distrai e decide sozinha quando agir; o acerto tende a ser menor |
| **Definição de "repouso"** | o repouso é a espera antes da seta, com a cruz na tela | a pessoa pode já estar se preparando; o modelo pode aprender "antes × depois do aviso", não "repouso × intenção" |
| **Resposta visual à seta** | a seta fica 1,25 s na tela; a janela de intenção começa em 1,5 s | a resposta do cérebro ao sumiço da seta dura algumas centenas de milissegundos e pode entrar no começo da janela; parte do acerto do EEGNet pode vir daí, e isso não foi medido |
| **Sinais que não são intenção** | piscadas, movimento dos olhos e atenção à tela aparecem no EEG | os eletrodos dos olhos ficaram fora, mas o efeito nos eletrodos da cabeça não foi medido |
| **Marcação de ruído** | feita pelos autores do conjunto, não refeita aqui | os testes de qualidade olham as tabelas, não o sinal: um eletrodo ruim passaria |
| **Quarentena desigual** | A06 perde 24,7% das tentativas; A01, 3,8% | o treino tem menos dados de quem tem mais ruído, e as classes deixam de ser exatamente equilibradas |
| **Regra diferente no teste** | treino e validação sem ruído; teste com tudo | escolha proposital (uso real); o acerto nos trechos com ruído é quase o mesmo (EEGNet: 75,5% × 77,0%) |

## 2. Limites de generalização

- **9 pessoas no total.** Cada pessoa é testada uma vez, por modelos treinados nas outras 7. A
  média (área 0,86 no EEGNet) resume 9 números, e eles variam muito (0,72 a 0,96): uma pessoa nova
  pode ficar fora dessa faixa.
- **Os testes estatísticos têm pouca força.** Com 9 pessoas, o menor p possível no Wilcoxon é 0,004;
  só diferenças que aparecem em quase todas as pessoas chegam a ser significativas. Diferenças
  pequenas (por movimento imaginado, por ruído) não seriam detectadas.
- **Uma semente.** Cada modelo foi treinado uma vez por rodada (semente 42). Outra semente daria
  números um pouco diferentes; essa variação não foi medida.
- **O corte 0,5 não serve para todos.** A área ROC não depende do corte, mas a acurácia sim: no
  CSP + LDA, o mesmo corte faz uma pessoa receber "intenção" em quase tudo e outra quase nunca. Uso
  real exigiria calibrar o corte em cada pessoa.
- **A nota do modelo não é probabilidade.** Não foi calibrada: 0,8 não quer dizer 80% de chance.
- **Só 2 dias por pessoa,** juntos no treino e no teste. Não dá para saber se o modelo continua
  valendo semanas depois.
- **Um equipamento, um laboratório.** 22 eletrodos, 250 Hz, mesma montagem. Outro equipamento não
  foi testado.
- **Embeddings congelados.** O CBraMod foi usado sem ajuste, com trechos de 1 s. O resultado mede
  o que o pré-treino já sabia, não o que o modelo consegue aprender.

## 3. O que não se pode concluir

| não se pode dizer que... | por quê |
|---|---|
| o modelo detecta intenção **motora** | ele separa "antes" de "depois" da seta; resposta visual, atenção e expectativa também mudam nesse intervalo, e não foi feita análise de quais eletrodos ou ritmos pesam na decisão |
| o método serve para pacientes ou para uso clínico | só há voluntários saudáveis, em laboratório, com o momento da tentativa marcado pela tela |
| o modelo funciona bem em qualquer pessoa nova | a área vai de 0,72 a 0,96 entre as 9; com tão poucas pessoas, a faixa real pode ser maior |
| o EEGNet é melhor que o CSP + LDA em geral | ganhou nas 9 pessoas, mas com uma configuração de cada (filtro, janela, semente); outra configuração do CSP + LDA não foi testada |
| o CBraMod não serve para esta tarefa | só foi testado congelado e com votação de vizinhos; com ajuste fino o resultado pode ser outro |
| não há vazamento nenhum | a pessoa de teste nunca entra no treino e não há trechos copiados; outros caminhos, como a ordem das tentativas, não foram testados |
| o sinal identifica uma pessoa | 25,4% dos vizinhos são da mesma pessoa (acaso: 11,1%): é um indício de "impressão digital", não um teste de reidentificação |
| o tipo de movimento ou o ruído não importam | a diferença foi pequena (82% a 86% das intenções detectadas por movimento), mas com 9 pessoas o teste tem pouca força para achar uma |

## 4. Reprodução

- **Mesmo ambiente, mesmo resultado.** Sementes fixas e CPU: duas execuções do zero deram os mesmos
  números. Entre versões de biblioteca, sistemas ou processadores diferentes, os pesos treinados
  podem mudar nas últimas casas e alterar algum número. Por isso os modelos, as predições e o
  registro da execução ficam no repositório.
- **Dados fora do repositório.** A licença (CC BY-ND 4.0) impede redistribuir o sinal e os vetores de
  embedding. A reprodução depende de o MOABB e o Hugging Face continuarem servindo os arquivos.
