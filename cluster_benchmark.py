import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def extract_triples_and_labels():
    """
    1. Extrai as triplas para o treinamento cheio do PyKeen.
    2. Extrai os rótulos de categoria (tag_type) de cada entidade para o clustering.
    """
    print("[INFO] Conectando ao Neo4j para extração de topologia e metadados...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query_triples = """
    MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
    WHERE r.tipo_num IS NOT NULL
    RETURN h.text AS head, r.tipo_num AS relation, t.text AS tail
    """
    
    query_labels = """
    MATCH (e:Entidade)
    RETURN e.text AS text, e.tag_type AS tag
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
    df_labels['text'] = df_labels['text'].astype(str)
    df_labels['tag'] = df_labels['tag'].astype(str)
    
    label_dict = dict(zip(df_labels['text'], df_labels['tag']))
    
    print(f"[SUCESSO] {len(df_triples)} triplas clínicas e {len(label_dict)} mapeamentos de categorias carregados.")
    return df_triples, label_dict

def process_and_evaluate_space(model_result, label_dict):
    """
    Passo 3 e Passo 4: Extrai os embeddings, aplica a conversão do módulo complexo 
    se necessário e calcula as métricas de validação de cluster reais.
    """
    model = model_result.model
    triples_factory = model_result.training
    
    # 1. Extrair os embeddings de entidades do modelo do PyKeen
    entity_representations = model.entity_representations[0]
    # Retorna o array de pesos (pode conter tensores complexos do PyTorch)
    raw_embeddings = entity_representations().cpu().detach().numpy()
    
    # PASSO 3: Equalização do Espaço Complexo via Módulo Absoluto
    if np.iscomplexobj(raw_embeddings) or 'complex' in str(raw_embeddings.dtype):
        # Transforma o plano complexo em vetor de magnitudes reais
        vector_matrix = np.abs(raw_embeddings)
    else:
        vector_matrix = raw_embeddings

    # Alinhar os rótulos clínicos com os IDs internos do PyKeen
    ordered_tags = []
    valid_indices = []
    
    for entity_text, entity_id in triples_factory.entity_to_id.items():
        # Busca a tag original do Neo4j para aquela palavra específica
        tag = label_dict.get(entity_text, "Unknown")
        # Ignora nós sem categoria ou não mapeados para não distorcer a estatística
        if tag != "Unknown" and tag != "None":
            ordered_tags.append(tag)
            valid_indices.append(entity_id)
            
    # Filtra as matrizes para conter apenas elementos com rótulos clínicos válidos
    X = vector_matrix[valid_indices]
    y = np.array(ordered_tags)
    
    # PASSO 4: Teste Estatístico de Coesão e Separação
    # Garante que temos múltiplos clusters válidos para calcular as métricas
    if len(set(y)) > 1:
        sil = silhouette_score(X, y, metric='cosine', sample_size=3000, random_state=42)
        db = davies_bouldin_score(X, y)
        return sil, db
    else:
        return 0.0, 0.0

def main():
    print("=================================================================")
    print("     BENCHMARK COMPARATIVO QUANTITATIVO: NODE CLUSTERING")
    print("=================================================================\n")
    
    try:
        # 1. Obter os dados (Passo 2)
        df_triples, label_dict = extract_triples_and_labels()
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df_triples[['head', 'relation', 'tail']].values
        )
        
        models_to_test = ['TransE', 'ComplEx', 'RotatE']
        comparison_table = []
        
        # 2. Execução dos Treinamentos Globais Cheios
        for model_name in models_to_test:
            print(f"\n[MODELO: {model_name}] Otimizando espaço latente e coletando métricas geométricas...")
            try:
                result = pipeline(
                    training=triples_factory,
                    testing=triples_factory,
                    model=model_name,
                    model_kwargs=dict(embedding_dim=64),
                    training_kwargs=dict(num_epochs=10, batch_size=256),
                    device='cpu'
                )
                
                # Executa a equalização e a validação estatística (Passos 3 e 4)
                sil, db = process_and_evaluate_space(result, label_dict)
                
                comparison_table.append({
                    'Modelo': model_name,
                    'Silhouette': sil,
                    'Davies-Bouldin': db
                })
                print(f"[SUCESSO] Avaliação espacial do {model_name} concluída.")
                
            except Exception as e_model:
                print(f"[ERRO NO MODELO {model_name}] Falha na clusterização: {e_model}")
                comparison_table.append({
                    'Modelo': model_name, 'Silhouette': 0.0, 'Davies-Bouldin': 0.0
                })

        # 3. Consolidação e Exibição da Tabela Final (Passo 5)
        print("\n" + "="*65)
        print("          TABELA COMPARATIVA DE CLUSTERIZAÇÃO SEMÂNTICA")
        print("="*65)
        print(f"{'Modelo':<15} | {'Silhouette Score (↑)':<22} | {'Davies-Bouldin (↓)':<20}")
        print("-"*65)
        for row in comparison_table:
            print(f"{row['Modelo']:<15} | {row['Silhouette']:<22.4f} | {row['Davies-Bouldin']:<20.4f}")
        print("="*65)
        print("\n[SUCESSO] Node clustering concluído!")
        
    except Exception as e:
        print(f"\n[ERRO GERAL] Falha no pipeline de clustering: {e}")

if __name__ == "__main__":
    main()