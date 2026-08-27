import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def extract_clean_triples():
    """
    Conecta ao Neo4j e extrai as triplas estruturadas, 
    garantindo que apenas arestas com tipo válido sejam lidas.
    """
    print("[INFO] Conectando ao Neo4j para extração de triplas clínicas...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
    WHERE r.tipo IS NOT NULL
    RETURN h.id AS head, r.tipo AS relation, t.id AS tail
    """
    
    with driver.session() as session:
        result = session.run(query)
        df = pd.DataFrame([dict(record) for record in result])
        
    driver.close()
    
    if df.empty:
        raise ValueError("[ERRO] Nenhuma tripla válida com 'tipo' foi encontrada no banco.")
        
    df['head'] = df['head'].astype(str)
    df['relation'] = df['relation'].astype(str)
    df['tail'] = df['tail'].astype(str)
    
    print(f"[SUCESSO] Extração concluída: {len(df)} triplas recuperadas.")
    return df

def get_clean_metrics(metric_results):
    """
    Extrai MRR e Hits@1/3/10 da variante 'both.realistic' do PyKeen,
    casando o nome da metrica pelo segmento EXATO da chave (ex.
    "both.realistic.hits_at_1"), nao por substring.

    A versao anterior usava `'hits_at_1' in k_lower`, que tambem da match
    em "hits_at_10" (que contem "hits_at_1" como substring!), fazendo
    Hits@1 ser sobrescrito pelo valor de Hits@10 sempre que essa chave
    aparecia depois na iteracao do dict — o sintoma era Hits@1 > Hits@3
    nos resultados, o que e matematicamente impossivel (se a resposta
    certa esta no top-1, ela necessariamente esta no top-3 tambem). O
    guard de "realistic" tambem era frouxo demais: `not any(x in k_lower
    for x in ['head', 'tail'])` aceitava por engano as variantes
    'optimistic'/'pessimistic' de 'both', nao so 'realistic'.
    """
    flat = metric_results.to_flat_dict()
    mrr, h1, h3, h10 = 0.0, 0.0, 0.0, 0.0

    for k, v in flat.items():
        parts = k.lower().split('.')
        # Exige 'both' e 'realistic' como segmentos exatos da chave (nao
        # substring), e casa o nome da metrica pelo ultimo segmento exato.
        if 'both' not in parts or 'realistic' not in parts:
            continue
        metric_name = parts[-1]
        if metric_name in ('inverse_harmonic_mean_rank', 'mrr'):
            mrr = v
        elif metric_name == 'hits_at_1':
            h1 = v
        elif metric_name == 'hits_at_3':
            h3 = v
        elif metric_name == 'hits_at_10':
            h10 = v

    if mrr == 0.0 and h1 == 0.0 and h3 == 0.0 and h10 == 0.0:
        # Nenhuma chave bateu com o formato esperado "both.realistic.<nome>" —
        # mais provavel indicar uma versao do PyKeen com nomenclatura
        # diferente do que verificar de novo por substring (o que reintroduz
        # o bug). Imprime as chaves cruas para diagnostico manual.
        print("[AVISO] Nenhuma metrica 'both.realistic.*' encontrada. Chaves disponiveis:")
        for k in sorted(flat.keys()):
            print(f"    {k}")

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