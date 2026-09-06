"""
Item 8 do plano de acao pos-banca: prototipo de RAG (Retrieval-Augmented
Generation) usando o grafo de conhecimento clinico como fonte de contexto.

Este script treina UMA VEZ o modelo de embeddings de melhor desempenho
identificado nos itens 4 e 5 (RotatE -- ver Tabela do link prediction,
Hits@10 muito acima dos outros dois modelos; e Tabela de clustering por
tag_type, melhor Davies-Bouldin e praticamente empatado em Silhouette com
o ComplEx) sobre TODO o grafo unificado, e salva os embeddings resultantes
em disco (results/rag_embeddings.npz e results/rag_entity_metadata.json).

Diferenca deliberada frente a cluster_benchmark.py/compare_models.py: ali
o objetivo e avaliar e comparar os tres modelos (por isso ha split
treino/teste e 5 seeds por modelo); aqui o objetivo e produzir UM artefato
de producao reutilizavel pelo prototipo de RAG (src/rag_retriever.py), sem
retreinar a cada pergunta. Os hiperparametros (embedding_dim=64, 10
epocas, batch_size=256) sao os mesmos do regime uniforme usado nos
benchmarks, para manter consistencia metodologica com o texto do Cap. 4
(secoes 4.2.4/4.2.5, itens 6/7); a seed fica fixa em 0 so para
reprodutibilidade do artefato, nao como comparacao entre execucoes.
"""
import json
import os

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

MODELO = "RotatE"
SEED = 0
OUTPUT_DIR = "results"
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "rag_embeddings.npz")
METADATA_PATH = os.path.join(OUTPUT_DIR, "rag_entity_metadata.json")


def extrair_triplas_e_metadados():
    """
    Extrai (1) as triplas completas do grafo, para treinar o modelo sobre
    toda a base (sem split -- este e o modelo de producao, nao uma rodada
    de avaliacao), e (2) os metadados de cada entidade (texto, categoria,
    ocorrencias) para exibicao no prototipo.
    """
    print("[INFO] Conectando ao Neo4j para extrair triplas e metadados...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query_triplas = """
    MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
    WHERE r.tipo IS NOT NULL
    RETURN h.id AS head, r.tipo AS relation, t.id AS tail
    """
    query_metadados = """
    MATCH (e:Entidade)
    RETURN e.id AS id, e.text AS text, e.tag_type AS tag_type,
           e.ocorrencias AS ocorrencias
    """

    with driver.session() as session:
        df_triplas = pd.DataFrame([dict(r) for r in session.run(query_triplas)])
        df_metadados = pd.DataFrame([dict(r) for r in session.run(query_metadados)])

    driver.close()

    if df_triplas.empty:
        raise ValueError("[ERRO] Nenhuma tripla encontrada no Neo4j. O grafo foi populado (main.py)?")

    df_triplas = df_triplas.astype(str)
    df_metadados["id"] = df_metadados["id"].astype(str)

    print(f"[SUCESSO] {len(df_triplas)} triplas e {len(df_metadados)} entidades carregadas.")
    return df_triplas, df_metadados


def treinar_rotate(df_triplas):
    """
    Treina o RotatE sobre o grafo completo (treino == teste, ja que aqui
    nao se mede desempenho -- so se produz o espaco latente final). Mesmos
    hiperparametros do benchmark (dim=64, 10 epocas, batch 256), seed fixa
    para reprodutibilidade do artefato.
    """
    triples_factory = TriplesFactory.from_labeled_triples(
        triples=df_triplas[["head", "relation", "tail"]].values
    )
    print(f"[INFO] Treinando {MODELO} sobre o grafo completo ({triples_factory.num_triples} triplas)...")
    result = pipeline(
        training=triples_factory,
        testing=triples_factory,
        model=MODELO,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=10, batch_size=256),
        random_seed=SEED,
        device="cpu",
    )
    return result


def extrair_espaco_normalizado(result):
    """
    Mesma extracao ja validada em cluster_benchmark.py (extract_normalized_space):
    concatena parte real e imaginaria do RotatE, preservando a fase (ver
    commit 6f61c57), e normaliza L2 (ver commit 79fa6c7). Retorna os ids
    das entidades (na ordem das linhas da matriz) e a matriz em si.
    """
    model = result.model
    triples_factory = result.training

    raw_embeddings = model.entity_representations[0]().cpu().detach().numpy()
    if np.iscomplexobj(raw_embeddings):
        vector_matrix = np.concatenate([raw_embeddings.real, raw_embeddings.imag], axis=1)
    else:
        vector_matrix = raw_embeddings

    norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vector_matrix = vector_matrix / norms

    id_to_index = triples_factory.entity_to_id
    ids_ordenados = [None] * len(id_to_index)
    for entity_label, idx in id_to_index.items():
        ids_ordenados[idx] = entity_label

    return np.array(ids_ordenados), vector_matrix


def main():
    print("=================================================================")
    print("  ITEM 8: TREINAMENTO DO MODELO DE PRODUCAO PARA O PROTOTIPO RAG")
    print(f"  Tecnica escolhida: {MODELO} (melhor desempenho nos itens 4/5)")
    print("=================================================================\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_triplas, df_metadados = extrair_triplas_e_metadados()
    result = treinar_rotate(df_triplas)
    ids_ordenados, matriz_embeddings = extrair_espaco_normalizado(result)

    np.savez_compressed(EMBEDDINGS_PATH, ids=ids_ordenados, embeddings=matriz_embeddings)
    print(
        f"[SUCESSO] Embeddings salvos em: {EMBEDDINGS_PATH} "
        f"({matriz_embeddings.shape[0]} entidades, dim={matriz_embeddings.shape[1]})"
    )

    metadados_dict = {
        row["id"]: {
            "text": row["text"],
            "tag_type": row["tag_type"],
            "ocorrencias": int(row["ocorrencias"]) if pd.notna(row["ocorrencias"]) else None,
        }
        for _, row in df_metadados.iterrows()
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadados_dict, f, ensure_ascii=False, indent=2)
    print(f"[SUCESSO] Metadados de {len(metadados_dict)} entidades salvos em: {METADATA_PATH}")

    print("\n[PRONTO] Agora rode: streamlit run app_rag.py")


if __name__ == "__main__":
    main()
