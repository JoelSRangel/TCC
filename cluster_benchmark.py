import json
import os
import statistics

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Numero de execucoes independentes por modelo (uma por seed) — mesmo
# motivo do compare_models.py: sem seed fixa, dois runs do mesmo modelo
# davam numeros diferentes, e as diferencas entre TransE/ComplEx/RotatE
# aqui sao pequenas o suficiente (na primeira rodada real, Silhouette
# variou so entre -0.317 e -0.349) para que ruido de inicializacao pudesse
# mudar o ranking. Reportamos media +/- desvio padrao.
SEEDS = [0, 1, 2, 3, 4]

# Valores de K testados no K-means real (item 5 do plano de acao). O valor
# igual ao numero de categorias tag_type distintas no corpus (ja
# normalizadas, ver normalize_tag() abaixo) e adicionado dinamicamente em
# main(), para servir de ponto de comparacao com a analise de
# separabilidade por categoria pre-existente (ver tag_type_metrics abaixo).
#
# K=2, 3 e 4 incluidos a pedido do orientador (ponto 3 do feedback da
# banca, que sugeriu testar "2, 3, ..., 10"): e nessa faixa baixa que um
# agrupamento natural apareceria, se existir. Sem esses valores, a
# afirmacao de que o espaco nao tem granularidade natural de agrupamento
# ficava apoiada so na faixa de K >= 5.
K_VALUES_BASE = [2, 3, 4, 5, 10, 15, 20, 25, 30]

# Marcadores de anotacao do SemClinBr que descrevem uma propriedade da
# mencao (forma abreviada, polaridade negada), nao uma categoria clinica em
# si. Quando aparecem combinados a um tipo semantico mais substantivo (ex.:
# "Abbreviation|Laboratory or Test Result"), sao descartados por
# normalize_tag() para que o rotulo usado no clustering reflita a
# categoria clinica da entidade, nao o marcador.
_TAG_MODIFIERS = {"Abbreviation", "Negation"}


def normalize_tag(raw_tag):
    """
    O atributo bruto 'tag' do XML do SemClinBr pode vir composto por
    multiplos tipos semanticos separados por '|' (ex.: uma substancia que e
    ao mesmo tempo "Organic Chemical" e "Pharmacologic Substance"), em
    ordem inconsistente entre anotacoes equivalentes (tanto
    "Finding|Abbreviation" quanto "Abbreviation|Finding" ocorrem no
    corpus). Usado sem normalizacao, produz ~922 rotulos distintos no
    corpus atual — nao 23 categorias, como se assumia antes de conferir os
    dados (ver plano de acao, secao "Item 5").

    Esta funcao reduz cada anotacao a um unico rotulo de categoria clinica,
    dos 89 tipos semanticos base observados no corpus:
      1. remove tags duplicadas dentro da mesma anotacao (ex.:
         "Negation|Negation" -> "Negation");
      2. descarta os marcadores _TAG_MODIFIERS da lista, mantendo o
         primeiro tipo semantico substantivo restante;
      3. se so sobrar marcador (ou a tag for so um marcador), usa o
         marcador como categoria residual, em vez de descartar a entidade.

    Validado contra os XMLs brutos do corpus: reduz as 922 combinacoes
    distintas para 85 categorias (dos 89 tipos semanticos base possiveis;
    os 4 que faltam — Amino Acid Sequence, Behavior, Plant, Vitamin — so
    aparecem combinados atras de outro tipo mais frequente na mesma
    anotacao, e acabam absorvidos por ele; afeta poucas dezenas de
    anotacoes no corpus, documentado como limitacao no texto).
    """
    parts = []
    for part in raw_tag.split("|"):
        if part not in parts:
            parts.append(part)

    substantive = [p for p in parts if p not in _TAG_MODIFIERS]
    if substantive:
        return substantive[0]
    return parts[0] if parts else raw_tag


def extract_triples_and_labels():
    """
    1. Extrai as triplas para o treinamento cheio do PyKeen.
    2. Extrai os rótulos de categoria (tag_type) de cada entidade para o
       clustering, normalizados por normalize_tag() (ver docstring acima).
    """
    print("[INFO] Conectando ao Neo4j para extração de topologia e metadados...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query_triples = """
    MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
    WHERE r.tipo IS NOT NULL
    RETURN h.id AS head, r.tipo AS relation, t.id AS tail
    """

    query_labels = """
    MATCH (e:Entidade)
    RETURN e.id AS id, e.tag_type AS tag
    """

    with driver.session() as session:
        res_triples = session.run(query_triples)
        df_triples = pd.DataFrame([dict(r) for r in res_triples])

        res_labels = session.run(query_labels)
        df_labels = pd.DataFrame([dict(r) for r in res_labels])

    driver.close()

    if df_triples.empty or df_labels.empty:
        raise ValueError("[ERRO] Falha ao recuperar triplas ou categorias do banco.")

    df_triples = df_triples.astype(str)
    df_labels['id'] = df_labels['id'].astype(str)
    df_labels['tag'] = df_labels['tag'].astype(str).apply(normalize_tag)

    label_dict = dict(zip(df_labels['id'], df_labels['tag']))

    print(f"[SUCESSO] {len(df_triples)} triplas clínicas e {len(label_dict)} mapeamentos de categorias carregados.")
    return df_triples, label_dict


def count_valid_tags(label_dict):
    """Numero de categorias tag_type distintas no corpus (ja normalizadas
    por normalize_tag()), excluindo os marcadores 'Unknown'/'None'. Usado
    so para incluir esse valor no sweep de K, como ponto de comparacao com
    a analise por tag_type."""
    tags = {t for t in label_dict.values() if t not in ("Unknown", "None")}
    return len(tags)


def extract_normalized_space(model_result, label_dict):
    """
    Extrai os embeddings do modelo treinado, equaliza o espaço complexo e
    normaliza para norma unitária. Retorna X (embeddings das entidades com
    tag_type conhecido) e y (os rótulos tag_type correspondentes, na mesma
    ordem de X) — usados tanto pela análise por categoria pré-existente
    (tag_type_metrics) quanto pelo K-means real (kmeans_sweep).

    A conversão do espaço complexo (ComplEx/RotatE usam embeddings de
    valor complexo; TransE já é real) concatena parte real e imaginária em
    vez de reduzir para módulo absoluto, para não descartar a fase do
    número complexo — ver histórico no commit 6f61c57 e na seção 4.2.5.
    """
    model = model_result.model
    triples_factory = model_result.training

    entity_representations = model.entity_representations[0]
    raw_embeddings = entity_representations().cpu().detach().numpy()

    if np.iscomplexobj(raw_embeddings) or 'complex' in str(raw_embeddings.dtype):
        vector_matrix = np.concatenate([raw_embeddings.real, raw_embeddings.imag], axis=1)
    else:
        vector_matrix = raw_embeddings

    ordered_tags = []
    valid_indices = []

    for entity_label, entity_id in triples_factory.entity_to_id.items():
        tag = label_dict.get(entity_label, "Unknown")
        if tag != "Unknown" and tag != "None":
            ordered_tags.append(tag)
            valid_indices.append(entity_id)

    X = vector_matrix[valid_indices]
    y = np.array(ordered_tags)

    # Normalização L2: ver histórico no commit 79fa6c7 e na seção 4.2.5 —
    # equaliza a geometria entre TransE (real) e ComplEx/RotatE (módulo
    # complexo) para o Davies-Bouldin, que sempre usa distância Euclidiana.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # evita divisão por zero em vetores nulos
    X_normalizado = X / norms

    return X_normalizado, y


def tag_type_metrics(X_normalizado, y):
    """
    Silhouette/Davies-Bouldin usando tag_type como rótulo — mede
    separabilidade por categoria semântica JÁ EXISTENTE no corpus, não por
    clusters descobertos. Mantido por continuidade com a rodada anterior
    (seção 4.2.5); o clustering de verdade é feito por kmeans_sweep.
    """
    if len(set(y)) > 1:
        sil = silhouette_score(X_normalizado, y, metric='cosine', sample_size=3000, random_state=42)
        db = davies_bouldin_score(X_normalizado, y)
        return sil, db
    return 0.0, 0.0


def kmeans_sweep(X_normalizado, k_values, seed):
    """
    Item 5 do plano de ação: clustering real via K-means, testando
    múltiplos valores de K. Diferente de tag_type_metrics(), os clusters
    aqui são descobertos pelo K-means a partir da geometria dos
    embeddings, sem usar tag_type em nenhum momento — só depois de
    formados os clusters é que dá pra comparar com as categorias reais
    (o que fica pra interpretação no texto, não neste script).
    """
    n_samples = X_normalizado.shape[0]
    results = {}
    for k in k_values:
        if k < 2 or k >= n_samples:
            continue
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_pred = kmeans.fit_predict(X_normalizado)
        if len(set(labels_pred)) < 2:
            continue
        sil = silhouette_score(X_normalizado, labels_pred, metric='cosine', sample_size=3000, random_state=42)
        db = davies_bouldin_score(X_normalizado, labels_pred)
        results[k] = {'silhouette': sil, 'davies_bouldin': db}
    return results


def run_single(model_name, triples_factory, label_dict, k_values, seed):
    result = pipeline(
        training=triples_factory,
        testing=triples_factory,
        model=model_name,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=10, batch_size=256),
        random_seed=seed,
        device='cpu',
    )
    X_normalizado, y = extract_normalized_space(result, label_dict)
    sil_tag, db_tag = tag_type_metrics(X_normalizado, y)
    kmeans_results = kmeans_sweep(X_normalizado, k_values, seed)
    return sil_tag, db_tag, kmeans_results


def summarize(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    print("=================================================================")
    print("     BENCHMARK COMPARATIVO QUANTITATIVO: NODE CLUSTERING")
    print(f"     ({len(SEEDS)} execucoes por modelo, seeds={SEEDS})")
    print("=================================================================")
    print("[INFO] Duas analises sao feitas: (1) separabilidade por categoria")
    print("tag_type ja existente no corpus, e (2) clustering real via")
    print("K-means, com K variavel (item 5 do plano de acao).\n")

    try:
        # 1. Obter os dados
        df_triples, label_dict = extract_triples_and_labels()
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df_triples[['head', 'relation', 'tail']].values
        )

        n_tags = count_valid_tags(label_dict)
        K_VALUES = sorted(set(K_VALUES_BASE + [n_tags]))
        print(f"[INFO] Categorias tag_type distintas no corpus (normalizadas, ver normalize_tag): {n_tags}.")
        print(f"[INFO] Valores de K testados no K-means: {K_VALUES}\n")

        models_to_test = ['TransE', 'ComplEx', 'RotatE']
        raw_runs = []
        summary_table = []

        # 2. Execução dos Treinamentos, N execuções por modelo
        for model_name in models_to_test:
            print(f"\n[MODELO: {model_name}] Otimizando espaço latente em {len(SEEDS)} execuções independentes...")
            sils, dbs = [], []
            kmeans_per_k = {k: {'silhouette': [], 'davies_bouldin': []} for k in K_VALUES}

            for seed in SEEDS:
                print(f"  [seed={seed}] treinando e avaliando espaço latente...")
                try:
                    sil, db, kmeans_results = run_single(model_name, triples_factory, label_dict, K_VALUES, seed)
                except Exception as e_seed:
                    print(f"  [ERRO seed={seed}] Falha nesta execução: {e_seed}")
                    sil, db, kmeans_results = 0.0, 0.0, {}
                print(f"    -> Silhouette(tag_type)={sil:.4f}  Davies-Bouldin(tag_type)={db:.4f}")

                sils.append(sil); dbs.append(db)
                raw_runs.append({
                    'modelo': model_name, 'seed': seed,
                    'silhouette_tag_type': sil, 'davies_bouldin_tag_type': db,
                    'kmeans': kmeans_results,
                })
                for k, metrics_k in kmeans_results.items():
                    kmeans_per_k[k]['silhouette'].append(metrics_k['silhouette'])
                    kmeans_per_k[k]['davies_bouldin'].append(metrics_k['davies_bouldin'])

            sil_mean, sil_std = summarize(sils)
            db_mean, db_std = summarize(dbs)

            kmeans_summary = {}
            for k in K_VALUES:
                sil_vals = kmeans_per_k[k]['silhouette']
                db_vals = kmeans_per_k[k]['davies_bouldin']
                if not sil_vals:
                    continue
                s_mean, s_std = summarize(sil_vals)
                d_mean, d_std = summarize(db_vals)
                kmeans_summary[k] = {
                    'silhouette_media': s_mean, 'silhouette_desvio': s_std,
                    'davies_bouldin_media': d_mean, 'davies_bouldin_desvio': d_std,
                }

            # K que maximiza o Silhouette medio entre os valores testados —
            # heuristica simples de escolha de K, reportada junto com o
            # sweep completo (nao substitui julgamento manual do "cotovelo").
            melhor_k = max(kmeans_summary, key=lambda k: kmeans_summary[k]['silhouette_media']) if kmeans_summary else None

            summary_table.append({
                'Modelo': model_name,
                'Silhouette_media': sil_mean, 'Silhouette_desvio': sil_std,
                'DaviesBouldin_media': db_mean, 'DaviesBouldin_desvio': db_std,
                'kmeans_sweep': kmeans_summary,
                'kmeans_melhor_k': melhor_k,
            })
            print(f"[SUCESSO] {model_name}: {len(SEEDS)} execuções concluídas.")

        # 3. Tabela de separabilidade por tag_type (mantida por continuidade)
        print("\n" + "="*80)
        print("  TABELA 1: SEPARABILIDADE POR CATEGORIA (tag_type) — média ± desvio, N=" + str(len(SEEDS)))
        print("="*80)
        print(f"{'Modelo':<10} | {'Silhouette (↑)':<22} | {'Davies-Bouldin (↓)':<22}")
        print("-"*80)
        for row in summary_table:
            print(
                f"{row['Modelo']:<10} | "
                f"{row['Silhouette_media']:.4f} ± {row['Silhouette_desvio']:.4f}      | "
                f"{row['DaviesBouldin_media']:.4f} ± {row['DaviesBouldin_desvio']:.4f}"
            )
        print("="*80)

        # 4. Tabela de K-means real, Silhouette por K testado
        print("\n" + "="*100)
        print("  TABELA 2: K-MEANS REAL — Silhouette (↑) médio por K, N=" + str(len(SEEDS)))
        print("="*100)
        header = f"{'Modelo':<10} | " + " | ".join(f"K={k:<4}" for k in K_VALUES)
        print(header)
        print("-"*100)
        for row in summary_table:
            cells = []
            for k in K_VALUES:
                entry = row['kmeans_sweep'].get(k)
                cells.append(f"{entry['silhouette_media']:.3f}" if entry else "  -   ")
            print(f"{row['Modelo']:<10} | " + " | ".join(f"{c:<6}" for c in cells))
        print("="*100)

        print("\n[K sugerido por modelo — maior Silhouette médio entre os K testados]")
        for row in summary_table:
            k_best = row['kmeans_melhor_k']
            if k_best is not None:
                best = row['kmeans_sweep'][k_best]
                print(
                    f"  {row['Modelo']:<10} -> K={k_best} "
                    f"(Silhouette={best['silhouette_media']:.4f} ± {best['silhouette_desvio']:.4f}, "
                    f"Davies-Bouldin={best['davies_bouldin_media']:.4f} ± {best['davies_bouldin_desvio']:.4f})"
                )
        print("\n[SUCESSO] Node clustering concluído!")

        # 5. Salva execuções individuais + resumo para auditoria/reprodutibilidade
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", "clustering_resultados.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seeds": SEEDS,
                    "k_values": K_VALUES,
                    "tags_distintas": n_tags,
                    "execucoes_individuais": raw_runs,
                    "resumo": summary_table,
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"[SUCESSO] Resultados detalhados (todas as execuções + resumo) salvos em: {output_path}")

    except Exception as e:
        print(f"\n[ERRO GERAL] Falha no pipeline de clustering: {e}")

if __name__ == "__main__":
    main()
