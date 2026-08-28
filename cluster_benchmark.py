import json
import os
import statistics

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Numero de execucoes independentes por modelo (uma por seed) — mesmo
# motivo do compare_models.py: sem seed fixa, dois runs do mesmo modelo
# davam numeros diferentes, e as diferencas entre TransE/ComplEx/RotatE
# aqui sao pequenas o suficiente (na primeira rodada real, Silhouette
# variou so entre -0.317 e -0.349) para que ruido de inicializacao pudesse
# mudar o ranking. Reportamos media +/- desvio padrao.
SEEDS = [0, 1, 2, 3, 4]


def extract_triples_and_labels():
    """
    1. Extrai as triplas para o treinamento cheio do PyKeen.
    2. Extrai os rótulos de categoria (tag_type) de cada entidade para o clustering.
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
    df_labels['tag'] = df_labels['tag'].astype(str)
    
    label_dict = dict(zip(df_labels['id'], df_labels['tag']))
    
    print(f"[SUCESSO] {len(df_triples)} triplas clínicas e {len(label_dict)} mapeamentos de categorias carregados.")
    return df_triples, label_dict


def process_and_evaluate_space(model_result, label_dict):
    """
    Extrai os embeddings, equaliza o espaço complexo (se necessário) e
    calcula Silhouette/Davies-Bouldin usando tag_type como rótulo.

    IMPORTANTE (limitação metodológica, ver item 5 do plano de ação): isto
    mede o quanto o espaço de embedding separa entidades pela categoria
    semântica JÁ EXISTENTE no corpus (tag_type), não por clusters
    descobertos via K-means. É uma métrica válida de "separabilidade por
    rótulo externo", mas não substitui um K-sweep de verdade.
    """
    model = model_result.model
    triples_factory = model_result.training
    
    entity_representations = model.entity_representations[0]
    raw_embeddings = entity_representations().cpu().detach().numpy()
    
    # Equalização do espaço complexo via módulo absoluto (ComplEx/RotatE
    # usam embeddings de valor complexo; TransE já é real). Isso descarta a
    # fase do número complexo — que é onde parte da expressividade desses
    # dois modelos vive (relações antissimétricas, por exemplo) — então a
    # comparação de clustering pode desfavorecer ComplEx/RotatE frente ao
    # TransE nativamente real. Registrar essa ressalva ao interpretar.
    if np.iscomplexobj(raw_embeddings) or 'complex' in str(raw_embeddings.dtype):
        vector_matrix = np.abs(raw_embeddings)
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

    # Normalização L2: silhouette_score(metric='cosine') já é invariante à
    # escala dos vetores, mas davies_bouldin_score do scikit-learn SEMPRE
    # usa distância Euclidiana (não aceita parâmetro de métrica) — sem
    # normalizar, essa métrica fica sensível à norma bruta dos embeddings,
    # que difere estruturalmente entre TransE (real) e ComplEx/RotatE
    # (módulo de número complexo), tornando a comparação entre modelos
    # não-equitativa. Normalizando cada vetor para norma 1 antes das duas
    # métricas, a distância Euclidiana entre vetores unitários vira uma
    # função monotônica da distância de cosseno
    # (||a-b||^2 = 2 - 2*cos(a,b)), então Davies-Bouldin passa a refletir
    # a mesma geometria do Silhouette, em vez de uma geometria diferente.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # evita divisão por zero em vetores nulos
    X_normalizado = X / norms
    
    if len(set(y)) > 1:
        sil = silhouette_score(X_normalizado, y, metric='cosine', sample_size=3000, random_state=42)
        db = davies_bouldin_score(X_normalizado, y)
        return sil, db
    else:
        return 0.0, 0.0


def run_single(model_name, triples_factory, label_dict, seed):
    result = pipeline(
        training=triples_factory,
        testing=triples_factory,
        model=model_name,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=10, batch_size=256),
        random_seed=seed,
        device='cpu',
    )
    return process_and_evaluate_space(result, label_dict)


def summarize(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    print("=================================================================")
    print("     BENCHMARK COMPARATIVO QUANTITATIVO: NODE CLUSTERING")
    print(f"     ({len(SEEDS)} execucoes por modelo, seeds={SEEDS})")
    print("=================================================================")
    print("[AVISO METODOLOGICO] Esta tabela mede separabilidade por categoria")
    print("semantica JA EXISTENTE (tag_type), nao clusters descobertos via")
    print("K-means. Ver item 5 do plano de acao para o K-sweep de verdade.\n")
    
    try:
        # 1. Obter os dados
        df_triples, label_dict = extract_triples_and_labels()
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df_triples[['head', 'relation', 'tail']].values
        )
        
        models_to_test = ['TransE', 'ComplEx', 'RotatE']
        raw_runs = []
        summary_table = []
        
        # 2. Execução dos Treinamentos, N execuções por modelo
        for model_name in models_to_test:
            print(f"\n[MODELO: {model_name}] Otimizando espaço latente em {len(SEEDS)} execuções independentes...")
            sils, dbs = [], []
            for seed in SEEDS:
                print(f"  [seed={seed}] treinando e avaliando espaço latente...")
                try:
                    sil, db = run_single(model_name, triples_factory, label_dict, seed)
                except Exception as e_seed:
                    print(f"  [ERRO seed={seed}] Falha nesta execução: {e_seed}")
                    sil, db = 0.0, 0.0
                print(f"    -> Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
                sils.append(sil); dbs.append(db)
                raw_runs.append({'modelo': model_name, 'seed': seed, 'silhouette': sil, 'davies_bouldin': db})

            sil_mean, sil_std = summarize(sils)
            db_mean, db_std = summarize(dbs)
            summary_table.append({
                'Modelo': model_name,
                'Silhouette_media': sil_mean, 'Silhouette_desvio': sil_std,
                'DaviesBouldin_media': db_mean, 'DaviesBouldin_desvio': db_std,
            })
            print(f"[SUCESSO] {model_name}: {len(SEEDS)} execuções concluídas.")

        # 3. Consolidação e Exibição da Tabela Final (média ± desvio padrão)
        print("\n" + "="*80)
        print("  TABELA COMPARATIVA DE CLUSTERIZAÇÃO SEMÂNTICA — média ± desvio, N=" + str(len(SEEDS)))
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
        print("\n[SUCESSO] Node clustering concluído!")

        # 4. Salva execuções individuais + resumo para auditoria/reprodutibilidade
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", "clustering_resultados.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"seeds": SEEDS, "execucoes_individuais": raw_runs, "resumo": summary_table},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[SUCESSO] Resultados detalhados (todas as execuções + resumo) salvos em: {output_path}")
        
    except Exception as e:
        print(f"\n[ERRO GERAL] Falha no pipeline de clustering: {e}")

if __name__ == "__main__":
    main()
