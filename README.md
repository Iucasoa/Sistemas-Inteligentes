# Lista de Projetos 1: Sistemas Inteligentes (PAM0466)

**Aluno:** João Lucas Oliveira de Andrade
**Docente:** Pedro Thiago Valério de Souza
**Instituição:** UFERSA, Centro Multidisciplinar Pau dos Ferros | Semestre 2026.2

## Projetos

| # | Problema | Algoritmo | Relatório | Código |
|---|----------|-----------|-----------|--------|
| 1 | Previsão de energia em usina de ciclo combinado | Regressão linear múltipla | [relatório](problema1-regressao-linear/README.md) | [notebook](problema1-regressao-linear/notebook.ipynb) |
| 2 | Previsão de doenças cardíacas | Regressão logística | [relatório](problema2-regressao-logistica/README.md) | [notebook](problema2-regressao-logistica/notebook.ipynb) |
| 3 | Qualidade de vinhos | k-NN | [relatório](problema3-knn/README.md) | [notebook](problema3-knn/notebook.ipynb) |
| 4 | Risco de crédito | Árvore de decisão | [relatório](problema4-arvore-decisao/README.md) | [notebook](problema4-arvore-decisao/notebook.ipynb) |
| 5 | Segmentação de clientes | k-Means | [relatório](problema5-kmeans/README.md) | [notebook](problema5-kmeans/notebook.ipynb) |

Os notebooks seguem a estrutura e as ferramentas dos notebooks da disciplina ([pedrothiag/intelligent-systems](https://github.com/pedrothiag/intelligent-systems)).

## Como executar

Cada pasta contém um notebook (`notebook.ipynb`), o relatório (`README.md`) e as figuras usadas no relatório (`figuras/`, geradas pelo próprio notebook).
Os notebooks podem ser abertos no Google Colab (botão no topo de cada um) ou executados localmente:

```bash
pip install -r requirements.txt
```

Os datasets são obtidos automaticamente da UCI Machine Learning Repository (ou do OpenML, no Problema 4) dentro de cada notebook.
