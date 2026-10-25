# Análises que geram os números citados no texto do TCC

Cada script recalcula, a partir dos arquivos do projeto, números que aparecem no texto do TCC. Na saída, ao lado de cada valor, aparece entre colchetes o número que está no texto, para facilitar a conferência.

Os scripts 01 a 06 usam só a biblioteca padrão do Python 3: não precisam do Neo4j, do PyKEEN nem do ambiente conda. O 07 precisa do PyKEEN (ambiente conda `tcc_graphs`), mas não do Neo4j.

| Script | O que calcula | Onde aparece no texto |
|---|---|---|
| `01_relacoes.py` | Tipos de relação, direção da negação, cardinalidade, composição e exemplos de triplas | Cap. 2 (SemClinBr, TransE, ComplEx, RotatE), cap. 3 (Base de Dados, Modelagem) e cap. 4 (limitações dos modelos KGE) |
| `02_categorias.py` | 922 combinações de categorias reduzidas a 85; arestas que ligam a mesma categoria (6,5%) | Cap. 3 (métricas de agrupamento) e cap. 4 (separação das categorias, análise exploratória) |
| `03_estrutura_do_grafo.py` | Estrutura do grafo por menção e do grafo unificado; distribuição de grau | Cap. 4, seção 4.1 |
| `04_referencia_ao_acaso.py` | MRR e Hits@K esperados ao acaso, comparados com os modelos | Cap. 3 (métricas de predição de links) e cap. 4 (tabela de predição de links) |
| `05_vizinhos_rag.py` | Similaridade com o vizinho mais próximo nos vetores do RotatE e em vetores aleatórios | Cap. 4, protótipo de RAG |
| `06_tabelas_dos_benchmarks.py` | Valores das tabelas de predição de links, categorias e K-means | Cap. 4 |
| `07_configuracao_pykeen.py` | Configuração que o PyKEEN de fato usou em cada modelo: perda, otimizador, taxa de aprendizado, amostragem negativa, regularização, restrições e avaliação | Cap. 3 (configuração dos modelos e hiperparâmetros) |

## Como rodar

Da raiz do repositório:

```
python analises/01_relacoes.py
python analises/02_categorias.py
python analises/03_estrutura_do_grafo.py
python analises/04_referencia_ao_acaso.py
python analises/05_vizinhos_rag.py
python analises/06_tabelas_dos_benchmarks.py
```

O `05` leva cerca de 20 segundos; os outros, poucos segundos.

O `07` é o único que precisa do PyKEEN, então roda no ambiente conda do projeto (no WSL) e leva cerca de 20 segundos:

```
conda activate tcc_graphs
python analises/07_configuracao_pykeen.py
```

## Arquivos de que dependem

- `data/raw/*.xml`: corpus SemClinBr (não versionado). Usado pelos scripts 01 a 04.
- `results/rag_embeddings.npz`: gerado por `treinar_embeddings_rag.py` (branch `feature/rag_prototype`). Usado pelo 05.
- `results/link_prediction_resultados.json`: gerado por `compare_models.py`. Usado pelo 04 e pelo 06.
- `results/clustering_resultados.json`: gerado por `cluster_benchmark.py`. Usado pelo 06.

Os grafos são montados com a mesma chave de `src/extractor.py`: texto normalizado (maiúsculas, sem espaços repetidos) mais a categoria bruta da anotação.
