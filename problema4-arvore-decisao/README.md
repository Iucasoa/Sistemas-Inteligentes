# Problema 4: Classificação de Risco de Crédito Utilizando Árvores de Decisão

**Disciplina:** PAM0466 Sistemas Inteligentes | UFERSA, Centro Multidisciplinar Pau dos Ferros | 2026.2
**Aluno:** João Lucas Oliveira de Andrade
**Código:** [notebook.ipynb](notebook.ipynb)

---

## 1. Introdução

A concessão de crédito é uma das decisões mais sensíveis do sistema financeiro: aprovar um cliente inadimplente gera prejuízo direto, mas recusar um bom pagador significa perda de receita e exclusão financeira. Aqui não basta acertar, é preciso **justificar a decisão**: a LGPD (Lei 13.709/2018), em seu Art. 20, assegura ao titular o direito de solicitar a revisão de decisões tomadas exclusivamente com base em tratamento automatizado, o que valoriza modelos interpretáveis.

A árvore de decisão atende a esse requisito: o modelo aprendido pode ser lido como um conjunto de **regras de produção** que um analista, um cliente ou um auditor compreendem sem conhecimento estatístico. Neste trabalho foi construída uma árvore para classificar solicitantes de crédito em **bom pagador** e **mau pagador**.

### Dataset

**Statlog (German Credit Data)** (UCI; Hans Hofmann, Universidade de Hamburgo, 1994): **1.000 solicitações de crédito** de um banco alemão, descritas por **20 atributos (7 numéricos e 13 categóricos)**. O rótulo indica bom risco (700 amostras) ou mau risco (300 amostras). Foi usada a versão com rótulos legíveis, `fetch_openml(name='credit-g', version=1)`.

### Fundamentação teórica

Cada nó interno da árvore aplica um teste $x_j \le t$: as amostras que respondem **sim** seguem para a esquerda e as que respondem **não**, para a direita. Para cada atributo, os limiares candidatos são os pontos médios entre valores consecutivos; escolhe-se a divisão de maior **ganho de impureza** (impureza do nó pai menos a impureza ponderada dos filhos). O nó vira folha quando é puro, quando tem menos amostras que o mínimo para dividir, quando atinge a profundidade máxima ou quando nenhuma divisão traz ganho; a folha devolve a **classe majoritária**. As medidas de impureza são:

$$\text{Gini}(D) = 1 - \sum_{i=1}^{c} p_i^2 \qquad\qquad \text{Entropia}(D) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

**Implementação.** O código segue o notebook de árvores da disciplina (`08_Tree.ipynb`): divisão estratificada com `random_state=42`, `DecisionTreeClassifier` com `criterion`, `max_depth`, `min_samples_split=2` e `min_samples_leaf=1`, `plot_tree`, `export_text`, `accuracy_score` e `confusion_matrix`. Do notebook de pré-processamento (`04_Preprocessamento.ipynb`) vêm o `OneHotEncoder`, o `DummyClassifier` e o SMOTE.

---

## 2. Tarefa 1: Preparação dos dados

### 2.1 Atributos numéricos e categóricos

O dataset tem dimensão **(1000, 20)**, **nenhum valor ausente** e nenhuma linha duplicada.

- **Numéricos (7):** `duration`, `credit_amount`, `installment_commitment`, `residence_since`, `age`, `existing_credits`, `num_dependents`.
- **Categóricos (13):** `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, `personal_status`, `other_parties`, `property_magnitude`, `other_payment_plans`, `housing`, `job`, `own_telephone`, `foreign_worker`.

| Atributo categórico | Nº de categorias |
|---|---|
| purpose | 10 |
| savings_status, credit_history, employment | 5 cada |
| checking_status, personal_status, property_magnitude, job | 4 cada |
| other_parties, other_payment_plans, housing | 3 cada |
| own_telephone, foreign_worker | 2 cada |

| Atributo numérico | Média | Desvio padrão | Mínimo | Máximo |
|---|---|---|---|---|
| duration (meses) | 20,90 | 12,06 | 4 | 72 |
| credit_amount | 3271,26 | 2822,74 | 250 | 18424 |
| installment_commitment | 2,97 | 1,12 | 1 | 4 |
| residence_since | 2,84 | 1,10 | 1 | 4 |
| age (anos) | 35,55 | 11,38 | 19 | 75 |
| existing_credits | 1,41 | 0,58 | 1 | 4 |
| num_dependents | 1,16 | 0,36 | 1 | 2 |

### 2.2 Codificação das variáveis categóricas

A `scikit-learn` exige entradas numéricas, então os atributos categóricos precisam ser codificados. Como visto na aula de pré-processamento, a **codificação ordinal** só é adequada para categorias **com ordem** natural; usá-la em atributos **nominais** (como finalidade do empréstimo ou tipo de moradia) cria uma **ordem falsa** que o modelo interpretaria como informação real. Por isso foi usado o ***one-hot encoding*** (`OneHotEncoder`), que cria uma coluna binária para cada categoria. O conjunto passou a ter **61 colunas** (7 numéricas e 54 binárias). Na árvore, um teste `<= 0,50` numa coluna binária significa que o cliente **não pertence** àquela categoria.

### 2.3 Balanceamento das classes e impacto sobre a acurácia

| Classe | Amostras | Proporção |
|---|---|---|
| 0 (bom pagador) | 700 | 70% |
| 1 (mau pagador) | 300 | 30% |

O conjunto é **desbalanceado (70% / 30%)**. Consequência: um classificador que prevê sempre "bom pagador" atinge **70% de acurácia sem ter aprendido nada**. No teste, o `DummyClassifier(strategy='most_frequent')` teve acurácia de **0,700** e **recall de 0,000** para o mau pagador: não identifica nenhum inadimplente. Portanto, a acurácia isolada **não é uma boa medida** neste problema: ela precisa ser comparada com esses 70% e acompanhada da matriz de confusão e das métricas da classe minoritária (mau pagador), que é a de maior interesse para o banco. Como se verá na Tarefa 4, esse problema afeta até a escolha da profundidade da árvore.

---

## 3. Tarefa 2: Por que a normalização não é necessária para árvores de decisão

Cada divisão da árvore compara **um único atributo** com um limiar, $x_j \le t$, e o limiar é escolhido pelo ganho de impureza, que depende apenas da **ordem** das amostras naquele atributo, nunca da sua magnitude nem de comparações entre atributos diferentes. Uma transformação estritamente crescente $g$ (como Min-Max ou Z-score) preserva a ordem, $x_a \le x_b \iff g(x_a) \le g(x_b)$: o limiar ótimo passa a ser $g(t)$ e as amostras vão para os mesmos filhos, então a árvore é equivalente. No k-NN, ao contrário, a distância euclidiana soma contribuições de todos os atributos, e os de maior escala (aqui, `credit_amount`, com desvio de 2.823, contra 0,36 de `num_dependents`) dominariam.

**Verificação experimental.** A árvore final foi treinada com os atributos numéricos originais e com eles padronizados: as **previsões no teste foram idênticas**. Por isso nenhuma normalização foi aplicada.

---

## 4. Tarefa 3: Divisão treino/teste estratificada

Os dados foram divididos em **80% para treino (800 amostras)** e **20% para teste (200 amostras)**, com `random_state=42` e **divisão estratificada** (`stratify=y`): **30% de maus pagadores tanto no treino (240 de 800) quanto no teste (60 de 200)**.

---

## 5. Tarefa 4: Controle de complexidade (profundidade máxima)

Para cada profundidade máxima (2, 3, 5, 7, 10 e sem limite) e cada critério (Gini e entropia), a árvore foi avaliada por **validação cruzada de 5 *folds* no treino** (`cross_validate(..., cv=5, return_train_score=True)`), registrando a acurácia de treino e de validação. Como a acurácia é enganosa com classes desbalanceadas, registrou-se também o **F1-score macro** (média simples do F1 das duas classes), que a aula de métricas indica como preferível nesse caso.

| Critério | Profundidade | Acurácia de treino | Acurácia de validação | F1 macro de treino | F1 macro de validação (± desvio) |
|---|---|---|---|---|---|
| Gini | 2 | 0,7075 | **0,7025** | 0,5057 | 0,5069 ± 0,1165 |
| Gini | 3 | 0,7378 | 0,6963 | 0,6670 | 0,6087 ± 0,0742 |
| Gini | 5 | 0,8187 | 0,6838 | 0,7738 | 0,6004 ± 0,0368 |
| Gini | 7 | 0,8841 | 0,6800 | 0,8564 | 0,6051 ± 0,0193 |
| Gini | 10 | 0,9597 | 0,6800 | 0,9509 | 0,6120 ± 0,0177 |
| Gini | sem limite | 1,0000 | 0,6600 | 1,0000 | 0,5947 ± 0,0239 |
| Entropia | 2 | 0,7034 | **0,7025** | 0,4591 | 0,4583 ± 0,0931 |
| Entropia | **3** | 0,7319 | 0,6925 | 0,6881 | **0,6458 ± 0,0175** |
| Entropia | 5 | 0,8000 | 0,6950 | 0,7608 | 0,6424 ± 0,0259 |
| Entropia | 7 | 0,8588 | 0,6525 | 0,8341 | 0,5934 ± 0,0128 |
| Entropia | 10 | 0,9363 | 0,6687 | 0,9226 | 0,6072 ± 0,0098 |
| Entropia | sem limite | 1,0000 | 0,6687 | 1,0000 | 0,6142 ± 0,0171 |

![Acurácia e F1 macro de treino e de validação em função da profundidade](figuras/curva_profundidade.png)

**A armadilha da acurácia.** Pela acurácia de validação, a melhor profundidade seria **2**, nos dois critérios (0,7025). Mas a árvore de profundidade 2 prevê **"bom pagador" para todos os 800 clientes do treino**: ela é idêntica ao classificador da classe majoritária, e sua acurácia de 70% não significa nada (o F1 macro de 0,51 revela isso). Por isso, **a profundidade foi escolhida pelo maior F1 macro de validação**, que só cresce quando o modelo acerta também os maus pagadores.

**Resposta: profundidade escolhida.** O maior F1 macro de validação foi o da **entropia com profundidade 3** (0,6458 ± 0,0175), escolhida para o modelo final. Com o Gini, o melhor F1 macro foi o da profundidade 10 (0,6120), praticamente empatado com a profundidade 3 (0,6087).

### O que acontece com profundidades muito baixas e muito altas?

- **Profundidade muito baixa (2): subajuste (*underfitting*), alto viés.** A árvore tem tão poucas regras que nenhuma folha chega a ter maioria de maus pagadores: ela não aprende o problema e só reproduz a classe majoritária. Treino e validação ficam iguais e ruins em F1 macro (0,51 e 0,51 no Gini).
- **Profundidade muito alta (sem limite): sobreajuste (*overfitting*), alta variância.** A árvore cresce até as folhas ficarem puras e atinge **1,0000** no treino, mas a validação cai (acurácia de 0,66 e F1 macro de 0,59 no Gini): ela "decora" ruídos do treino. A diferença entre treino e validação cresce continuamente com a profundidade (em acurácia, de 0,005 na profundidade 2 para 0,34 sem limite).
- **Faixa intermediária (3 a 5):** melhor equilíbrio; com a entropia, F1 macro de validação de 0,646 e 0,642.

### Gini versus entropia

Com profundidades altas os dois critérios se comportam de forma parecida (diferenças de até 0,02 no F1 macro, dentro do desvio entre os *folds*). A diferença mais visível aparece nas profundidades **3 e 5**, em que a entropia teve F1 macro cerca de 4 pontos percentuais maior (0,6458 contra 0,6087 na profundidade 3) e com desvio bem menor (0,018 contra 0,074). Houve, portanto, uma **diferença moderada a favor da entropia** nas árvores rasas, mas ela não é grande frente à variação entre os *folds* do Gini; os dois critérios medem a impureza de forma semelhante e costumam gerar árvores parecidas.

---

## 6. Tarefa 5: Interpretação do modelo

A árvore final (entropia, profundidade 3) tem **8 folhas**, o que a torna fácil de ler por inteiro.

![Árvore de decisão](figuras/arvore_decisao.png)

### 6.1 Importância dos atributos

A importância é a redução total de impureza trazida pelas divisões que usam cada atributo (normalizada para somar 1). Como o *one-hot* divide cada atributo categórico em várias colunas, as importâncias das colunas foram também somadas por atributo original:

| Coluna | Importância | | Atributo original | Importância |
|---|---|---|---|---|
| `checking_status_no checking` | 0,5153 | | **checking_status** | **0,5153** |
| `duration` | 0,1277 | | **duration** | **0,1277** |
| `other_payment_plans_bank` | 0,1022 | | **other_payment_plans** | **0,1022** |
| `property_magnitude_real estate` | 0,0992 | | property_magnitude | 0,0992 |
| `savings_status_<100` | 0,0778 | | savings_status | 0,0778 |
| `purpose_used car` | 0,0475 | | purpose | 0,0475 |
| `employment_4<=X<7` | 0,0302 | | employment | 0,0302 |

**Resposta:** os três atributos mais determinantes são **`checking_status`** (situação da conta corrente; especificamente, não ter conta corrente), **`duration`** (prazo do empréstimo) e **`other_payment_plans`** (ter outros financiamentos em banco), que somam cerca de **75%** da importância. `property_magnitude` vem logo em seguida (0,099), praticamente empatado com o terceiro. A importância **não informa o sentido** do efeito, apenas o quanto o atributo foi usado para reduzir a impureza.

### 6.2 Regra da folha com mais maus pagadores

A folha com o **maior número de amostras de mau pagador** (no treino) tem **141 maus e 127 bons** (52,6% de maus) e é a **única folha da árvore classificada como "mau"**. O caminho da raiz até ela é:

```
checking_status_no checking <= 0.50
duration > 11.50
savings_status_<100 > 0.50
```

**Em linguagem natural:**

> **SE** o cliente **possui conta corrente** (não está na categoria "sem conta corrente")
> **E** o **prazo** do empréstimo é **maior que 11,5 meses**
> **E** o cliente tem **poupança inferior a 100** marcos
> **ENTÃO** ele é classificado como **mau pagador**.

A regra é coerente do ponto de vista de risco: prazo longo e pouca reserva financeira. Ela também mostra que a folha é impura (quase metade dos clientes que caem nela são bons pagadores), o que antecipa os falsos positivos da Tarefa 6.

---

## 7. Tarefa 6: Avaliação no conjunto de teste

### 7.1 Matriz de confusão e métricas

![Matriz de confusão](figuras/matriz_confusao.png)

| Real \ Previsto | Bom pagador (0) | Mau pagador (1) |
|---|---|---|
| **Bom pagador (0)** | 103 (VN) | 37 (FP) |
| **Mau pagador (1)** | 26 (FN) | 34 (VP) |

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---|---|---|---|
| Bom pagador (0) | 0,80 | 0,74 | 0,77 | 140 |
| Mau pagador (1) | 0,48 | 0,57 | 0,52 | 60 |
| **Acurácia** | | | **0,685** | 200 |
| Média macro | 0,64 | 0,65 | 0,64 | 200 |
| Média ponderada | 0,70 | 0,69 | 0,69 | 200 |

A acurácia (0,685) é um pouco **menor** que a do classificador ingênuo (0,70), mas os dois não são comparáveis por esse número: o ingênuo não identifica nenhum mau pagador, enquanto a árvore identifica **34 dos 60** (recall de 0,57), e o F1 macro sobe de 0,41 para 0,64.

### 7.2 Qual tipo de erro é mais frequente e por que é o mais grave?

**Resposta:** o erro **mais frequente** foi o **falso positivo**, recusar um bom pagador (**37 casos**), contra **26 falsos negativos**, aprovar um mau pagador. Ainda assim, o erro **mais grave** é o **falso negativo**: conceder crédito a quem não vai pagar gera perda do capital emprestado, enquanto recusar um bom pagador gera "apenas" perda de receita (juros não recebidos) e exclusão do cliente. A própria documentação do dataset (`german.doc`) define uma matriz de custos em que classificar um mau pagador como bom custa **5** e o erro oposto custa **1**. Com esses pesos, os 26 falsos negativos custam 130, contra 37 dos falsos positivos. No teste, o modelo deixou passar **26 dos 60 maus pagadores (43%)**.

### 7.3 Ajuste no treinamento para reduzir o erro mais grave

O ajuste proposto é **tratar o desbalanceamento no treinamento**, para que a árvore dê mais peso à classe minoritária. Foram testados dois ajustes, mantendo o critério e a profundidade escolhidos por validação cruzada:

1. **SMOTE** (aula de pré-processamento): cria amostras sintéticas de maus pagadores por interpolação entre vizinhos, aplicado **somente no treino** (de 560/240 para 560/560).
2. **Ponderação de classes** (`class_weight='balanced'`): os erros na classe minoritária pesam mais no cálculo da impureza.

| Modelo | Acurácia | Recall (mau) | Maus aprovados (FN) | Bons recusados (FP) | Custo (5·FN + FP) |
|---|---|---|---|---|---|
| Original | 0,6850 | 0,57 | 26 | 37 | 167 |
| SMOTE | 0,6550 | 0,65 | 21 | 48 | 153 |
| `class_weight='balanced'` | 0,6050 | **0,87** | **8** | 71 | **111** |

Os dois ajustes reduzem os falsos negativos à custa de mais falsos positivos e de menor acurácia. A ponderação de classes foi a mais efetiva: reduziu os maus pagadores aprovados de **26 para 8** e o custo total de 167 para 111. Uma ressalva sobre o SMOTE: como ele interpola entre amostras, cria valores fracionários nas colunas *one-hot* (por exemplo, 0,4 de uma categoria), que não correspondem a clientes reais; a ponderação de classes não tem esse problema. Essa comparação usa o teste apenas para ilustrar o efeito; para adotar um dos ajustes, a escolha deveria ser feita por validação cruzada no treino.

---

## 8. Limitações

- **Desempenho modesto:** F1 macro de 0,64 e recall de 0,57 para o mau pagador no modelo sem ajuste.
- **Escolha de profundidade pouco decisiva:** várias configurações têm F1 macro dentro do desvio entre os *folds*.
- **Conjunto de teste pequeno:** 200 amostras (60 maus pagadores) e uma única divisão treino/teste (`random_state=42`).
- **Instabilidade das árvores:** pequenas variações nos dados podem alterar a estrutura da árvore e as regras.
- **Dataset pequeno e de origem única** (um banco alemão, década de 1990), com atributos sensíveis como estado civil/sexo (`personal_status`) e idade, cujo uso em decisões reais exige avaliação jurídica e de equidade.

---

## 9. Conclusão

1. **Preparação:** 1.000 solicitações (7 atributos numéricos e 13 categóricos), sem valores ausentes, codificadas com *one-hot* em 61 colunas. As classes são desbalanceadas (70%/30%): prever sempre "bom pagador" dá 70% de acurácia com recall zero para o mau pagador.
2. **Normalização:** desnecessária para árvores (invariância a transformações monotônicas), confirmada experimentalmente.
3. **Complexidade:** pela acurácia, a validação cruzada escolheria uma árvore de profundidade 2 que só prevê "bom pagador" (subajuste mascarado pelo desbalanceamento); pelo F1 macro, a escolhida foi a **entropia com profundidade 3** (0,6458). Árvores sem limite chegam a 100% no treino com validação menor (sobreajuste). A entropia foi moderadamente melhor que o Gini nas árvores rasas.
4. **Interpretação:** os atributos mais determinantes são `checking_status`, `duration` e `other_payment_plans`. A regra que leva à folha de mau risco é: possui conta corrente, prazo maior que 11,5 meses e poupança inferior a 100.
5. **Avaliação:** acurácia de 0,685, recall de 0,57 para o mau pagador; o erro mais frequente é o falso positivo (37), mas o mais grave é o falso negativo (26). A ponderação de classes reduziu os falsos negativos para 8.

---

## 10. Reprodutibilidade

- Código: [notebook.ipynb](notebook.ipynb), executável no Google Colab ou localmente (`pip install -r requirements.txt` na raiz do repositório).
- Dataset: UCI Machine Learning Repository, *Statlog (German Credit Data)*, obtido por `fetch_openml(name='credit-g', version=1)`.
- Semente aleatória: `random_state=42` (divisão treino/teste, árvores e SMOTE).

---

## 11. Referências

1. SOUZA, P. T. V. *Sistemas Inteligentes*: slides 04 (Pré-processamento), 05 (Métricas de Performance) e 09 (Árvores de Decisão) e notebooks `04_Preprocessamento.ipynb` e `08_Tree.ipynb`. UFERSA, 2026. Disponível em: https://github.com/pedrothiag/intelligent-systems.
2. HOFMANN, H. *Statlog (German Credit Data)*. UCI Machine Learning Repository, 1994.
