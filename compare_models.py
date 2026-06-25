import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def extract_clean_triples():
    """
    Conecta ao Neo4j e extrai as triplas estruturadas, 
    garantindo que apenas arestas com tipo_num válido sejam lidas.
    """
    print("[INFO] Conectando ao Neo4j para extração de triplas clínicas...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
    WHERE r.tipo_num IS NOT NULL
    RETURN h.text AS head, r.tipo_num AS relation, t.text AS tail
    """
    
    with driver.session() as session:
        result = session.run(query)
        df = pd.DataFrame([dict(record) for record in result])
        
    driver.close()
    
    if df.empty:
        raise ValueError("[ERRO] Nenhuma tripla válida com 'tipo_num' foi encontrada no banco.")
        
    df['head'] = df['head'].astype(str)
    df['relation'] = df['relation'].astype(str)
    df['tail'] = df['tail'].astype(str)
    
    print(f"[SUCESSO] Extração concluída: {len(df)} triplas recuperadas.")
    return df

def get_clean_metrics(metric_results):
    """
    Extrai de forma robusta as principais métricas do avaliador do PyKeen,
    procurando pelas chaves combinadas correspondentes à métrica 'realistic'.
    """
    flat = metric_results.to_flat_dict()
    mrr, h1, h3, h10 = 0.0, 0.0, 0.0, 0.0
    
    for k, v in flat.items():
        k_lower = k.lower()
        if 'inverse_harmonic_mean_rank' in k_lower or 'mrr' in k_lower:
            if 'both.realistic' in k_lower or not any(x in k_lower for x in ['head', 'tail']):
                mrr = v
        elif 'hits_at_1' in k_lower or ('hits_at_k' in k_lower and k_lower.endswith('.1')):
            if 'both.realistic' in k_lower or not any(x in k_lower for x in ['head', 'tail']):
                h1 = v
        elif 'hits_at_3' in k_lower or ('hits_at_k' in k_lower and k_lower.endswith('.3')):
            if 'both.realistic' in k_lower or not any(x in k_lower for x in ['head', 'tail']):
                h3 = v
        elif 'hits_at_10' in k_lower or ('hits_at_k' in k_lower and k_lower.endswith('.10')):
            if 'both.realistic' in k_lower or not any(x in k_lower for x in ['head', 'tail']):
                h10 = v
                
    # Fallback de segurança caso a busca estrita 'realistic' venha zerada
    if mrr == 0.0:
        for k, v in flat.items():
            if 'mrr' in k.lower() or 'inverse_harmonic_mean_rank' in k.lower():
                mrr = v
                break
    if h1 == 0.0:
        for k, v in flat.items():
            if 'hits_at_1' in k.lower() or 'hits_at_k_1' in k.lower():
                h1 = v
                break
    if h3 == 0.0:
        for k, v in flat.items():
            if 'hits_at_3' in k.lower() or 'hits_at_k_3' in k.lower():
                h3 = v
                break
    if h10 == 0.0:
        for k, v in flat.items():
            if 'hits_at_10' in k.lower() or 'hits_at_k_10' in k.lower():
                h10 = v
                break
                
    return mrr, h1, h3, h10

def main():
    print("=================================================================")
    print("     BENCHMARK COMPARATIVO QUANTITATIVO: LINK PREDICTION ")
    print("=================================================================\n")
    
    try:
        # 1. Obter e dividir os dados do Neo4j
        df = extract_clean_triples()
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df[['head', 'relation', 'tail']].values
        )
        
        print("\n[INFO] Aplicando divisão estatística via PyKeen (80% / 20%)...")
        training_factory, testing_factory = triples_factory.split([0.8, 0.2], random_state=42)
        
        models_to_test = ['TransE', 'ComplEx', 'RotatE']
        comparison_table = []
        
        # 2. Loop de Treinamento e Avaliação Real
        for model_name in models_to_test:
            print(f"\n[MODELO: {model_name}] Treinando nas triplas visíveis e avaliando no teste oculto...")
            try:
                # Argumento problemático removido! Adicionado dicionário padrão universal.
                result = pipeline(
                    training=training_factory,
                    testing=testing_factory,
                    model=model_name,
                    model_kwargs=dict(embedding_dim=64),
                    training_kwargs=dict(num_epochs=10, batch_size=256),
                    device='cpu'
                )
                
                # Extrai as métricas tratadas
                mrr, h1, h3, h10 = get_clean_metrics(result.metric_results)
                
                comparison_table.append({
                    'Modelo': model_name,
                    'MRR': mrr,
                    'Hits@1': h1,
                    'Hits@3': h3,
                    'Hits@10': h10
                })
                print(f"[SUCESSO] Avaliação do {model_name} concluída.")
                
            except Exception as e_model:
                print(f"[ERRO NO MODELO {model_name}] Falha na execução: {e_model}")
                comparison_table.append({
                    'Modelo': model_name,
                    'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0
                })

        # 3. Consolidação e Exibição da Tabela Final
        print("\n" + "="*70)
        print("          TABELA COMPARATIVA DE LINK PREDICTION (SEMCLINBR)")
        print("="*70)
        print(f"{'Modelo':<12} | {'MRR (↑)':<10} | {'Hits@1 (↑)':<10} | {'Hits@3 (↑)':<10} | {'Hits@10 (↑)':<10}")
        print("-"*70)
        for row in comparison_table:
            print(f"{row['Modelo']:<12} | {row['MRR']:<10.4f} | {row['Hits@1']:<10.4f} | {row['Hits@3']:<10.4f} | {row['Hits@10']:<10.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERRO GERAL] Falha no pipeline: {e}")

if __name__ == "__main__":
    main()