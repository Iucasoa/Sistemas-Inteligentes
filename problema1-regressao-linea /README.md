# Problema 1: Regressão Linear para Previsão de Produção de Energia em uma Usina de Ciclo Combinado

**Disciplina:** PAM0466 Sistemas Inteligentes | UFERSA, Centro Multidisciplinar Pau dos Ferros | 2026.2
**Aluno:** João Lucas Oliveira de Andrade
**Código:** [notebook.ipynb](notebook.ipynb)

---

## 1. Introdução

Uma usina termoelétrica de ciclo combinado (UTCC) combina uma turbina a gás (ciclo de Brayton) e uma turbina a vapor (ciclo de Rankine): o calor dos gases de exaustão da primeira produz o vapor que aciona a segunda, elevando o rendimento da usina (que pode chegar a cerca de 60%). O principal desafio técnico é integrar e sincronizar os dois ciclos, e por isso a potência elétrica de saída (PE) é muito sensível às condições ambientais.

Neste trabalho, construiu-se um modelo de **regressão linear múltipla** para prever PE a partir de quatro variáveis medidas por sensores ambientais, com o objetivo de apoiar a otimização do despacho energético.

### Dataset

Foi utilizado o dataset **UCI Combined Cycle Power Plant (CCPP)**, com 9.568 amostras coletadas ao longo de 6 anos, carregado diretamente do repositório da UCI pelo pacote `ucimlrepo` (id 294).

| Variável | Descrição | Unidade | Papel |
|---|---|---|---|
| AT | Temperatura ambiente | °C | Entrada |
| V | Vácuo de exaustão | cm Hg | Entrada |
| AP | Pressão atmosférica | mbar | Entrada |
| RH | Umidade relativa | % | Entrada |
| PE | Energia elétrica líquida horária de saída | MW | **Alvo** |

---

## 2. Pré-processamento

### 2.1 Inspeção e valores ausentes

O dataset tem dimensão **(9568, 5)**. A verificação de valores ausentes resultou em **zero valores faltantes em todas as colunas**, portanto nenhuma linha foi removida pelo `dropna()`.

Estatísticas descritivas:

| Variável | Média | Desvio padrão | Mínimo | Máximo |
|---|---|---|---|---|
| AT (°C) | 19,65 | 7,45 | 1,81 | 37,11 |
| V (cm Hg) | 54,31 | 12,71 | 25,36 | 81,56 |
| AP (mbar) | 1013,26 | 5,94 | 992,89 | 1033,30 |
| RH (%) | 73,31 | 14,60 | 25,56 | 100,16 |
| PE (MW) | 454,37 | 17,07 | 420,26 | 495,76 |

Os valores estão em faixas fisicamente plausíveis. As variáveis têm **escalas muito diferentes** (por exemplo, AP com média de ~1013 e desvio de ~5,9, contra RH com média de ~73 e desvio de ~14,6), o que motiva a padronização.

### 2.2 Divisão treino/teste e padronização

- Os dados foram divididos em **80% para treino (7.654 amostras) e 20% para teste (1.914 amostras)**, por sorteio aleatório simples com `random_state=42` para reprodutibilidade.
- As variáveis independentes foram padronizadas por **z-score**, $z = (x-\mu)/\sigma$, com `StandardScaler` ajustado **apenas no conjunto de treino** e depois aplicado ao teste. Essa ordem evita *data leakage*, isto é, o vazamento de informação do conjunto de teste para o treinamento.

---

## 3. Tarefa 1: Correlação entre as variáveis independentes e PE

Foi calculado o coeficiente de correlação de Pearson entre cada variável independente e PE:

| Variável | Correlação com PE |
|---|---|
| AT | **−0,948** |
| V | **−0,870** |
| AP | +0,518 |
| RH | +0,390 |

![Matriz de correlação de Pearson](figuras/matriz_correlacao.png)

**Resposta:** as variáveis com maior influência sobre a produção de energia são **AT** (r = −0,948) e **V** (r = −0,870), ambas com correlação forte e negativa: quanto maior a temperatura ambiente e o vácuo de exaustão, menor a potência gerada. AT é a mais influente. AP (r = +0,518) apresenta correlação moderada e positiva, e RH (r = +0,390) correlação mais fraca, também positiva.

O sentido físico é coerente: temperatura ambiente alta reduz a densidade do ar admitido pela turbina a gás, diminuindo o rendimento da combustão e a potência; o vácuo de exaustão está relacionado à pressão na turbina a vapor, e valores maiores indicam pior desempenho do ciclo de Rankine. Vale lembrar que correlação mede apenas relação **linear** e não implica causalidade.

---

## 4. Tarefa 2: Correlação entre as variáveis independentes (multicolinearidade)

Correlação de Pearson entre as variáveis independentes:

| | AT | V | AP | RH |
|---|---|---|---|---|
| **AT** | 1,000 | **0,844** | −0,508 | −0,543 |
| **V** | **0,844** | 1,000 | −0,414 | −0,312 |
| **AP** | −0,508 | −0,414 | 1,000 | 0,100 |
| **RH** | −0,543 | −0,312 | 0,100 | 1,000 |

Para quantificar a redundância entre as variáveis, calculou-se o **Fator de Inflação da Variância (VIF)**, definido como $VIF = 1/(1-R^2)$, em que $R^2$ vem da regressão de cada variável contra as demais:

| Variável | VIF |
|---|---|
| AT | **5,98** |
| V | 3,94 |
| AP | 1,45 |
| RH | 1,71 |

**Resposta:** sim, existe relação forte entre as variáveis independentes. O par **AT–V** tem correlação de **+0,844**, e AT também se correlaciona moderadamente com AP (−0,508) e RH (−0,543). O **VIF de AT (≈ 5,98) ultrapassa o limite usual de 5**, o que caracteriza **multicolinearidade moderada**; nenhum VIF chega a 10, portanto não se trata de um caso severo.

É plausível, do ponto de vista físico, que temperatura ambiente e vácuo de exaustão estejam acoplados, já que a temperatura do ar influencia as condições de condensação do vapor. A principal consequência da multicolinearidade é sobre a **interpretação dos coeficientes individuais** (que se tornam instáveis e difíceis de atribuir a uma só variável), e não tanto sobre a capacidade de previsão do modelo. Esse efeito aparece de forma concreta na Tarefa 3.

---

## 5. Tarefa 3: Modelo de regressão linear múltipla

O modelo ajustado tem a forma

$$PE = \theta_0 + \theta_1\,AT + \theta_2\,V + \theta_3\,AP + \theta_4\,RH$$

e foi treinado com `LinearRegression` do scikit-learn (mínimos quadrados ordinários) sobre os dados de treino padronizados.

**Coeficientes (variáveis padronizadas, em z-score):**

| Variável | Coeficiente |
|---|---|
| AT | −14,7639 |
| V | −2,9496 |
| AP | +0,3698 |
| RH | −2,3122 |

Intercepto (escala z-score): θ₀ = **454,4310** (corresponde à média de PE no conjunto de treino, como esperado quando as variáveis são padronizadas).

**Equação na escala original (°C, cm Hg, mbar, %):**

$$PE = 454{,}5691 - 1{,}9859\,AT - 0{,}2321\,V + 0{,}0622\,AP - 0{,}1581\,RH$$

| Variável | Coeficiente (escala original) |
|---|---|
| AT | −1,9859 MW/°C |
| V | −0,2321 MW/cm Hg |
| AP | +0,0622 MW/mbar |
| RH | −0,1581 MW/% |

**Interpretação na escala original:** mantendo as demais variáveis constantes, cada aumento de 1 °C na temperatura ambiente reduz a potência prevista em cerca de 1,99 MW; cada aumento de 1 cm Hg no vácuo de exaustão reduz cerca de 0,23 MW; cada 1 mbar a mais de pressão aumenta cerca de 0,06 MW; e cada ponto percentual a mais de umidade relativa reduz cerca de 0,16 MW. Como as variáveis têm unidades e escalas diferentes, esses coeficientes **não são comparáveis entre si** (o coeficiente pequeno de AP, por exemplo, reflete que 1 mbar é uma variação mínima diante do desvio padrão de ~5,9 mbar). Para comparar a importância das variáveis, usam-se os coeficientes padronizados, nos quais AT se destaca. O intercepto, nessa escala, não tem interpretação física direta, pois corresponde a AT = V = RH = AP = 0, fora da faixa observada dos dados.

**Análise dos coeficientes:** como as variáveis foram padronizadas, os coeficientes são comparáveis entre si. **AT tem de longe o maior peso** no modelo (|θ| = 14,76), seguida de V e RH; AP tem peso pequeno.

Um ponto importante é a **inversão de sinal em RH**: a correlação simples de RH com PE é **positiva (+0,390)**, mas o seu coeficiente no modelo é **negativo (−2,312)**. Isso ocorre porque RH é correlacionada com AT (r = −0,543): ao ajustar o modelo, a contribuição de RH passa a refletir apenas o efeito que sobra depois de descontar AT e V, enquanto a correlação simples mistura o efeito próprio de RH com o efeito de AT. Algo parecido acontece com AP, que tem correlação de +0,518 com PE mas coeficiente pequeno, porque parte da sua associação com PE é explicada por AT e V.

Isso mostra que a correlação isolada não basta para julgar a importância de uma variável e que, havendo multicolinearidade, **os coeficientes individuais devem ser interpretados com cautela**.

---

## 6. Tarefa 4: Avaliação do desempenho

As métricas foram calculadas sobre o **conjunto de teste** (dados não vistos no treinamento):

| Métrica | Valor |
|---|---|
| R² | **0,9301** |
| RMSE | **4,5026 MW** |
| MAE | **3,5959 MW** |

O RMSE equivale a diferentes percentuais conforme a referência:

| Referência | Valor | RMSE relativo |
|---|---|---|
| Faixa de PE (420,26 a 495,76 MW) | ≈ 75,5 MW | ≈ 6,0% |
| Média de PE | ≈ 454,4 MW | ≈ 1,0% |
| Desvio padrão de PE | ≈ 17,07 MW | ≈ 26% |

O modelo explica cerca de **93% da variância** de PE. O erro típico (RMSE) é de ~4,5 MW e o erro médio absoluto (MAE) é de ~3,6 MW; o RMSE ser um pouco maior que o MAE indica a presença de alguns erros maiores ocasionais.

### Diagnóstico dos resíduos

![Diagnóstico do modelo](figuras/diagnostico_modelo.png)

- **Valores reais vs. previstos:** os pontos se concentram em torno da diagonal, com um leve achatamento nas extremidades (tendência a subestimar os valores mais altos de PE e superestimar os mais baixos).
- **Distribuição dos resíduos:** aproximadamente simétrica, em forma de sino e **centrada em zero**, com uma cauda leve à esquerda (poucos erros grandes, chegando a cerca de −28 MW).
- **Resíduos vs. previstos:** não há formato de funil, o que indica **variância aproximadamente constante**. Observa-se, porém, uma **leve curvatura**: resíduos positivos nas extremidades e mais negativos na região central.

**Resposta:** o modelo é **adequado** para prever a produção de energia. O R² de 0,93 e o erro de poucos megawatts (cerca de 1% da produção média e 6% da faixa de operação) mostram boa capacidade preditiva, e os resíduos centrados em zero, simétricos e sem sinal de heterocedasticidade são compatíveis com as hipóteses da regressão linear.

Há, no entanto, **limitações**: a leve curvatura nos resíduos sugere uma pequena não linearidade na relação entre as variáveis e PE, que o modelo linear não captura; e a multicolinearidade moderada entre AT e V compromete a interpretação individual dos coeficientes, como evidenciado pela inversão de sinal de RH, embora afete pouco a previsão. Como possíveis melhorias, que **não foram testadas neste trabalho**, citam-se termos polinomiais, interações entre variáveis, regularização (Ridge/Lasso) ou modelos não lineares.

---

## 7. Conclusão

1. **Correlação com PE:** AT (−0,948) e V (−0,870) são as variáveis mais associadas à produção de energia, ambas de forma negativa; AP (+0,518) e RH (+0,390) têm associação menor.
2. **Multicolinearidade:** AT e V são fortemente correlacionadas (0,844) e o VIF de AT (≈ 5,98) indica multicolinearidade moderada.
3. **Modelo:** AT é a variável de maior peso (coeficiente padronizado −14,76); a inversão de sinal de RH ilustra a dificuldade de interpretar coeficientes individuais sob multicolinearidade.
4. **Desempenho:** R² = 0,9301, RMSE = 4,50 MW e MAE = 3,60 MW no conjunto de teste; o modelo é adequado, com as ressalvas de uma leve não linearidade e da multicolinearidade entre AT e V.

---

## 8. Reprodutibilidade

- Código: [notebook.ipynb](notebook.ipynb), executável no Google Colab.
- Dataset: UCI Machine Learning Repository, *Combined Cycle Power Plant* (id 294), obtido via `ucimlrepo`.
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `ucimlrepo`.
- Semente aleatória da divisão treino/teste: `random_state=42`.

