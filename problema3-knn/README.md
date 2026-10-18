# Problema 3: Classificação de Qualidade de Vinhos Utilizando k-NN

**Disciplina:** PAM0466 Sistemas Inteligentes | UFERSA, Centro Multidisciplinar Pau dos Ferros | 2026.2
**Aluno:** João Lucas Oliveira de Andrade
**Código:** [notebook.ipynb](notebook.ipynb)

---

## 1. Introdução

A avaliação de qualidade de vinhos depende historicamente de avaliações humanas, que são caras, lentas e subjetivas. Um modelo preditivo de qualidade tem aplicações no controle de lotes, na precificação e na certificação de denominação de origem, e ajuda a identificar quais variáveis químicas mais influenciam a percepção de qualidade.

Neste trabalho, construiu-se um classificador **k-NN (k-Nearest Neighbors)** que atribui cada vinho a uma de **três categorias de qualidade** (ruim, médio e bom) a partir de seus atributos físico-químicos.

### Dataset

**Wine Quality, arquivo `winequality-red.csv`** (Cortez et al., 2009 [1]; UCI): amostras de vinho tinto *Vinho Verde* (noroeste de Portugal), com **11 atributos físico-químicos** e um rótulo de qualidade inteiro de 0 a 10, atribuído por *sommeliers* (mediana de pelo menos três avaliações cegas). O arquivo foi carregado diretamente da UCI e tem dimensão **(1599, 12)**: 1.599 amostras, 11 atributos e a nota de qualidade.

Atributos: acidez fixa, acidez volátil, ácido cítrico, açúcar residual, cloretos, dióxido de enxofre livre, dióxido de enxofre total, densidade, pH, sulfatos e teor alcoólico.

### Fundamentação teórica do k-NN

O k-NN é um algoritmo supervisionado **não paramétrico** e do tipo ***lazy learner*** (aprendiz preguiçoso) [2]: diferentemente de modelos como a regressão logística ou as árvores de decisão, **não constrói um modelo matemático explícito** na fase de treinamento.

- **Treinamento:** consiste apenas em armazenar o conjunto de treino $TR = \{(\mathbf{x}^{(1)}, y^{(1)}), \dots, (\mathbf{x}^{(N)}, y^{(N)})\}$.
- **Predição:** o custo computacional é deslocado para a inferência. Para cada nova amostra $\mathbf{x}^*$: (1) calcula-se a distância até todas as $N$ amostras de treino; (2) selecionam-se as $k$ mais próximas, $\mathcal{N}_k(\mathbf{x}^*)$; (3) a classe é decidida por **votação majoritária** entre esses vizinhos:

$$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}^{(i)} \in \mathcal{N}_k(\mathbf{x}^*)} \mathbb{I}\big(y^{(i)} = c\big)$$

em que $\mathbb{I}(\cdot)$ vale 1 quando a condição é verdadeira e 0 caso contrário. A noção de "proximidade" é definida pela distância euclidiana (seção 2.3), o que torna o método sensível à escala dos atributos.

---

## 2. Tarefa 1: Preparação dos dados

### 2.1 Valores ausentes

Não há **nenhum valor ausente** em nenhuma das 12 colunas, portanto não foi necessário tratamento.

### 2.2 Recodificação da qualidade

A distribuição original das notas é:

| Nota | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Amostras | 10 | 53 | 681 | 638 | 199 | 18 |

A variável `quality` foi recodificada em três classes:

| Classe | Regra | Amostras | Proporção |
|---|---|---|---|
| ruim | nota ≤ 5 | 744 | 46,53% |
| médio | nota 6 ou 7 | 837 | 52,35% |
| bom | nota ≥ 8 | 18 | 1,13% |

O conjunto é **fortemente desbalanceado**: a classe "bom" tem apenas 18 amostras (cerca de 1% do total). Isso tem consequências diretas sobre a divisão, a validação cruzada e, principalmente, as métricas dessa classe, discutidas adiante.

### 2.3 Escalas dos atributos e normalização

Os atributos têm escalas muito diferentes:

| Atributo | Mínimo | Máximo | Desvio padrão |
|---|---|---|---|
| total sulfur dioxide | 6 | 289 | 32,90 |
| density | 0,99007 | 1,00369 | 0,001887 |
| volatile acidity | 0,12 | 1,58 | 0,179 |

O desvio padrão do dióxido de enxofre total é cerca de 17 mil vezes o da densidade.

Os atributos foram normalizados por **Z-score**, $z = (x-\mu)/\sigma$, com `StandardScaler` ajustado **apenas no treino** (e, na validação cruzada, apenas nas partições de treino de cada *fold*, por meio de um `Pipeline`), para evitar *data leakage*.

**Justificativa de que a normalização é obrigatória para o k-NN.** O k-NN classifica um ponto pelos seus vizinhos mais próximos, e "próximo" é definido pela distância euclidiana:

$$d(\mathbf{p},\mathbf{q}) = \sqrt{\sum_{j=1}^{11}(p_j-q_j)^2}$$

Sem normalização, os atributos de maior escala numérica dominam essa soma: uma diferença de dezenas em dióxido de enxofre total contribui com um termo milhões de vezes maior que uma diferença de milésimos em densidade, e os atributos de escala pequena praticamente deixam de influenciar quem são os vizinhos. Isso distorce a geometria do problema por uma questão de **unidade de medida**, e não de relevância química. A padronização coloca todos os atributos com média 0 e desvio 1, de modo que cada um contribua de forma comparável para a distância.

**Verificação experimental.** Com validação cruzada de 5 *folds* no conjunto de treino e k = 11, a acurácia média foi de **0,6216 sem normalização** e **0,7115 com normalização**, um ganho de cerca de 9 pontos percentuais, o que confirma o efeito da escala sobre o desempenho.

---

## 3. Tarefa 2: Divisão treino/teste estratificada

Os dados foram divididos em **80% para treino (1.279 amostras)** e **20% para teste (320 amostras)**, com `random_state=42` e **divisão estratificada** (`stratify=y`), que mantém a proporção das três classes nos dois conjuntos.

| Conjunto | ruim | médio | bom | Total |
|---|---|---|---|---|
| Treino | 595 | 670 | 14 | 1.279 |
| Teste | 149 | 167 | 4 | 320 |

A estratificação é especialmente importante aqui porque a classe "bom" tem poucas amostras: sem ela, um sorteio simples poderia deixar o teste sem nenhum exemplo dessa classe. Mesmo com ela, restaram **apenas 4 amostras "bom" no teste e 14 no treino**.

**Referência (baseline):** um classificador que sempre prevê a classe majoritária ("médio") acertaria **52,38%** das amostras do treino. A acurácia do k-NN deve ser lida em relação a esse valor.

---

## 4. Tarefa 3: Treinamento e escolha do k

Para cada valor de k, o modelo foi avaliado por **validação cruzada estratificada de 5 *folds* sobre o conjunto de treino**, com normalização ajustada dentro de cada *fold*. Foram testados k = 1, 3, 5, 7, 9, 11, 15, 21, 31 e 51.

| k | Acurácia de treino | Acurácia de validação (média) | Desvio entre folds |
|---|---|---|---|
| 1 | 1,0000 | **0,7248** | 0,0214 |
| 3 | 0,8362 | 0,6888 | 0,0175 |
| 5 | 0,7871 | 0,7013 | 0,0207 |
| 7 | 0,7766 | 0,7060 | 0,0332 |
| 9 | 0,7631 | 0,7170 | 0,0339 |
| 11 | 0,7617 | 0,7115 | 0,0361 |
| 15 | 0,7508 | 0,7193 | 0,0300 |
| 21 | 0,7441 | 0,7138 | 0,0332 |
| 31 | 0,7303 | 0,7146 | 0,0299 |
| 51 | 0,7275 | 0,7162 | 0,0188 |

![Acurácia em função de k](figuras/curva_k.png)

**Resposta: valor de k escolhido.** Pelo critério de maior acurácia média de validação, o valor escolhido foi **k = 1** (0,7248). **Essa escolha é pouco decisiva:** para os valores de k a partir de 9, a diferença é de menos de um ponto percentual (por exemplo, k = 15 teve 0,7193 e k = 51 teve 0,7162), muito menor que o desvio entre os *folds* (de 0,02 a 0,04). Do ponto de vista estatístico, esses valores de k (de 9 em diante) são praticamente equivalentes entre si e a k = 1. Os valores de k = 3, 5 e 7 tiveram acurácia de validação menor (de 0,6888 a 0,7060).

Um resultado que merece cautela: k = 1 costuma ser o mais sujeito a sobreajuste, e aqui foi o de maior acurácia de validação. Uma **hipótese plausível** é a presença de **amostras duplicadas** no dataset: as linhas 0 e 4 da tabela exibida na inspeção inicial são idênticas. Se amostras idênticas caírem em *folds* diferentes, o vizinho mais próximo de uma amostra de validação pode ser sua própria cópia no treino, com o mesmo rótulo, o que favorece k = 1. Essa hipótese **não foi testada** neste trabalho (seria necessário contar e remover os duplicados e repetir o experimento).

### O que acontece com k muito baixo e muito alto?

- **k muito baixo (k = 1):** cada ponto é classificado pelo seu único vizinho mais próximo, e o modelo acompanha qualquer ruído do treino. A acurácia de treino é **1,0000** (por construção, o vizinho mais próximo de um ponto de treino é ele mesmo), contra **0,7248** na validação: uma diferença de cerca de 27,5 pontos percentuais, característica de **sobreajuste (overfitting)**, com alta variância. A acurácia de validação em k = 3 (0,6888) é a mais baixa de todas, o que mostra que a curva não é monotônica.
- **k mais alto:** a diferença entre treino e validação diminui de forma consistente (de 0,275 em k = 1 para 0,011 em k = 51), ou seja, o modelo fica mais suave e menos ajustado a casos individuais. **Na faixa testada (até k = 51), a acurácia de validação não caiu** (0,7162 em k = 51), então o **subajuste (underfitting) não foi observado**. Em teoria, com k muito grande (da ordem do tamanho do treino) a votação seria dominada pela classe majoritária e a acurácia tenderia ao baseline (cerca de 52%); isso **não foi testado** aqui.

---

## 5. Tarefa 4: Avaliação no conjunto de teste

O modelo final (k = 1, com normalização ajustada no treino) foi treinado em todo o conjunto de treino e avaliado no conjunto de teste, obtendo **acurácia de 0,7656** (245 acertos em 320), contra o baseline de 0,5238, ou seja, cerca de 24 pontos percentuais acima de sempre prever "médio". Como a acurácia é estimada em 320 amostras, o erro-padrão aproximado é de cerca de 2,4 pontos percentuais ($\sqrt{p(1-p)/n}$ com $p=0{,}7656$ e $n=320$).

### 5.1 Matriz de confusão

![Matriz de confusão](figuras/matriz_confusao.png)

| Real \ Previsto | ruim | médio | bom |
|---|---|---|---|
| **ruim** | 112 | 37 | 0 |
| **médio** | 33 | 133 | 1 |
| **bom** | 0 | 4 | 0 |

### 5.2 Métricas por classe

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---|---|---|---|
| ruim | 0,77 | 0,75 | 0,76 | 149 |
| médio | 0,76 | 0,80 | 0,78 | 167 |
| bom | 0,00 | 0,00 | 0,00 | 4 |
| **Acurácia global** | | | **0,77** | 320 |
| Média macro | 0,51 | 0,52 | 0,51 | 320 |
| Média ponderada | 0,76 | 0,77 | 0,76 | 320 |

Definições (para cada classe, tratada como "positiva" contra as demais):

$$\text{Precisão}=\frac{VP}{VP+FP},\qquad \text{Recall}=\frac{VP}{VP+FN},\qquad F1=2\cdot\frac{\text{Precisão}\cdot\text{Recall}}{\text{Precisão}+\text{Recall}}$$

### 5.3 Entre quais classes o modelo erra com mais frequência?

**Resposta:** entre **ruim e médio**. Dos **75 erros** do teste, **70 (93%)** são confusões entre essas duas classes: 37 vinhos "ruim" previstos como "médio" e 33 vinhos "médio" previstos como "ruim". Os outros 5 erros envolvem a classe "bom".

Uma explicação plausível (hipótese, não testada) é que ruim e médio são classes **vizinhas na escala de qualidade**, separadas pela fronteira entre as notas 5 e 6, e que vinhos próximos a essa fronteira têm perfis químicos muito parecidos, o que dificulta a separação por distância. Notas 5 e 6 são, além disso, as mais frequentes do dataset (681 e 638 amostras).

### 5.4 A classe "bom"

O modelo **não acertou nenhum** dos 4 vinhos "bom" do teste (recall = 0): classificou os 4 como "médio" e, além disso, previu "bom" para 1 vinho que era "médio" (precisão = 0). O F1 da classe é 0.

Esse resultado mostra um limite importante da acurácia global: ela chega a 0,77 mesmo com a classe "bom" completamente errada, porque essa classe representa só 1,25% do teste. O contraste entre a **média ponderada** (F1 de 0,76) e a **média macro** (0,51) evidencia o problema. Com apenas 4 amostras no teste (e 14 no treino), as métricas dessa classe são muito instáveis (cada amostra muda o recall em 25 pontos percentuais) e **não permitem concluir que o modelo "não saiba" identificar vinhos bons**; apenas que, com tão poucos exemplos, o k-NN não os recupera. Como possibilidades **não testadas**, citam-se a ponderação dos vizinhos pela distância e técnicas de balanceamento.

---

## 6. Limitações

- **Classe minoritária ("bom"):** com 18 amostras no total, as estimativas de desempenho para essa classe têm pouca confiabilidade, tanto na validação cruzada quanto no teste (4 amostras).
- **Possíveis amostras duplicadas:** indícios de linhas idênticas no dataset podem inflar a acurácia, em especial para k = 1 e na avaliação no teste, se cópias de uma amostra aparecerem nos dois conjuntos. Não foi quantificado neste trabalho.
- **Escolha de k pouco decisiva:** as diferenças entre os valores de k estão dentro do desvio entre *folds*.
- **Uma única divisão treino/teste** (`random_state=42`): as métricas podem variar com outro sorteio.
- **Rótulos subjetivos:** a qualidade vem de avaliação sensorial humana, o que impõe um limite à precisão alcançável a partir da química.
- **Faixa de k testada:** como k só foi testado até 51, o comportamento para k muito grande (subajuste) não foi observado.

---

## 7. Conclusão

1. **Preparação:** o dataset tem 1.599 amostras, sem valores ausentes. A recodificação resultou em 744 vinhos "ruim", 837 "médio" e 18 "bom" (1,13%). A normalização por Z-score é necessária para o k-NN (distância euclidiana), e o experimento mostrou acurácia de validação de 0,6216 sem normalização contra 0,7115 com normalização (k = 11). A divisão estratificada deixou 1.279 amostras no treino e 320 no teste.
2. **Escolha de k:** pela validação cruzada de 5 *folds*, o melhor valor foi k = 1 (acurácia de 0,7248), com diferença pequena em relação aos demais e dentro do desvio entre *folds*. Observou-se sobreajuste em k baixo (treino de 1,0000 contra validação de 0,7248), e a diferença entre treino e validação diminuiu com k maior; o subajuste não foi observado até k = 51.
3. **Avaliação:** acurácia de 0,7656 no teste (baseline de 0,5238). O modelo erra principalmente entre as classes **ruim e médio** (70 dos 75 erros) e **não identificou nenhum dos 4 vinhos "bom"**, evidenciando que a acurácia global esconde o desempenho nulo na classe minoritária.

---

## 8. Reprodutibilidade

- Código: [notebook.ipynb](notebook.ipynb), executável no Google Colab.
- Dataset: UCI Machine Learning Repository, *Wine Quality* (`winequality-red.csv`), carregado por URL.
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Semente aleatória: `random_state=42` (divisão treino/teste e validação cruzada).

---

## 9. Referências

1. CORTEZ, P.; CERDEIRA, A.; ALMEIDA, F.; MATOS, T.; REIS, J. *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, v. 47, n. 4, p. 547-553, 2009.
2. FACELI, K.; LORENA, A. C.; GAMA, J.; ALMEIDA, T. A.; CARVALHO, A. C. P. L. F. *Inteligência Artificial: Uma Abordagem de Aprendizado de Máquina*. LTC. **[PREENCHER: conferir edição e ano na bibliografia indicada pelo professor.]**

