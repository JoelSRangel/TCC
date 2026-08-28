import json
import os
import statistics

import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Numero de execucoes independentes por modelo (uma por seed). Sem isso, a
# comparacao entre TransE/ComplEx/RotatE dependeria de uma unica
# inicializacao aleatoria do treinamento (o split treino/teste ja e fixo via
# random_state=42, mas o treinamento em si nao era — dois runs seguidos
# davam numeros diferentes). Reportamos media +/- desvio padrao sobre as
# SEEDS abaixo, e guardamos cada execucao individual no JSON de saida para
# auditoria.
SEEDS = [0, 1, 2, 3, 4]


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
    "both.realistic.hits_at_1"), nao por substring — ver historico no
    commit 1d2866f para o bug antigo (Hits@1 sendo contaminado por
    Hits@10 via match de substring).
    """
    flat = metric_results.to_flat_dict()
    mrr, h1, h3, h10 = 0.0, 0.0, 0.0, 0.0

    for k, v in flat.items():
        parts = k.lower().split('.')
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
        print("[AVISO] Nenhuma metrica 'both.realistic.*' encontrada. Chaves disponiveis:")
        for k in sorted(flat.keys()):
            print(f"    {k}")

    return mrr, h1, h3, h10


def run_single(model_name, training_factory, testing_factory, seed):
    """
    Roda o pipeline do PyKeen uma vez, com random_seed fixo, para uma
    execucao reprodutivel. evaluation_kwargs=dict(batch_size=256) evita que
    o PyKeen caia no batch size automatico e conservador (32) que ele usava
    por padrao na etapa de avaliacao — o log antigo mostrava esse aviso
    ("Using automatic batch size... conservative small batch_size=32"),
    entao o batch_size=256 pedido no training_kwargs so valia para o
    treino, nao para a avaliacao. Agora os dois ficam explicitos e iguais.
    """
    result = pipeline(
        training=training_factory,
        testing=testing_factory,
        model=model_name,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=10, batch_size=256),
        evaluation_kwargs=dict(batch_size=256),
        random_seed=seed,
        device='cpu',
    )
    return get_clean_metrics(result.metric_results)


def summarize(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    print("=================================================================")
    print("     BENCHMARK COMPARATIVO QUANTITATIVO: LINK PREDICTION ")
    print(f"     ({len(SEEDS)} execucoes por modelo, seeds={SEEDS})")
    print("=================================================================\n")
    
    try:
        # 1. Obter e dividir os dados do Neo4j
        df = extract_clean_triples()
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df[['head', 'relation', 'tail']].values
        )
        
        print("\n[INFO] Aplicando divisão estatística via PyKeen (80% / 20%, seed fixa 42)...")
        training_factory, testing_factory = triples_factory.split([0.8, 0.2], random_state=42)
        
        models_to_test = ['TransE', 'ComplEx', 'RotatE']
        raw_runs = []
        summary_table = []
        
        # 2. Loop de Treinamento e Avaliação Real, N execuções por modelo
        for model_name in models_to_test:
            print(f"\n[MODELO: {model_name}] Treinando e avaliando em {len(SEEDS)} execuções independentes...")
            mrrs, h1s, h3s, h10s = [], [], [], []
            for seed in SEEDS:
                print(f"  [seed={seed}] treinando nas triplas visíveis e avaliando no teste oculto...")
                try:
                    mrr, h1, h3, h10 = run_single(model_name, training_factory, testing_factory, seed)
                except Exception as e_seed:
                    print(f"  [ERRO seed={seed}] Falha nesta execução: {e_seed}")
                    mrr, h1, h3, h10 = 0.0, 0.0, 0.0, 0.0
                print(f"    -> MRR={mrr:.4f}  Hits@1={h1:.4f}  Hits@3={h3:.4f}  Hits@10={h10:.4f}")
                mrrs.append(mrr); h1s.append(h1); h3s.append(h3); h10s.append(h10)
                raw_runs.append({
                    'modelo': model_name, 'seed': seed,
                    'mrr': mrr, 'hits_at_1': h1, 'hits_at_3': h3, 'hits_at_10': h10,
                })

            mrr_mean, mrr_std = summarize(mrrs)
            h1_mean, h1_std = summarize(h1s)
            h3_mean, h3_std = summarize(h3s)
            h10_mean, h10_std = summarize(h10s)

            # Checagem de sanidade: se cada execucao individual satisfaz
            # Hits@1 <= Hits@3 <= Hits@10 (deve satisfazer sempre, por
            # definicao — top-1 correto implica top-3 correto), a media
            # tambem satisfaz. Alerta caso contrario (sinal de bug).
            if not (h1_mean <= h3_mean + 1e-9 <= h10_mean + 1e-9):
                print(f"  [AVISO] Inconsistencia entre medias de Hits@k para {model_name} "
                      f"(H1={h1_mean:.4f} H3={h3_mean:.4f} H10={h10_mean:.4f}) — checar bug de extracao.")

            summary_table.append({
                'Modelo': model_name,
                'MRR_media': mrr_mean, 'MRR_desvio': mrr_std,
                'Hits@1_media': h1_mean, 'Hits@1_desvio': h1_std,
                'Hits@3_media': h3_mean, 'Hits@3_desvio': h3_std,
                'Hits@10_media': h10_mean, 'Hits@10_desvio': h10_std,
            })
            print(f"[SUCESSO] {model_name}: {len(SEEDS)} execuções concluídas.")

        # 3. Consolidação e Exibição da Tabela Final (média ± desvio padrão)
        print("\n" + "="*90)
        print("     TABELA COMPARATIVA DE LINK PREDICTION (SEMCLINBR) — média ± desvio, N=" + str(len(SEEDS)))
        print("="*90)
        print(f"{'Modelo':<10} | {'MRR':<16} | {'Hits@1':<16} | {'Hits@3':<16} | {'Hits@10':<16}")
        print("-"*90)
        for row in summary_table:
            print(
                f"{row['Modelo']:<10} | "
                f"{row['MRR_media']:.4f} ± {row['MRR_desvio']:.4f}   | "
                f"{row['Hits@1_media']:.4f} ± {row['Hits@1_desvio']:.4f}   | "
                f"{row['Hits@3_media']:.4f} ± {row['Hits@3_desvio']:.4f}   | "
                f"{row['Hits@10_media']:.4f} ± {row['Hits@10_desvio']:.4f}"
            )
        print("="*90)

        # 4. Salva execuções individuais + resumo para auditoria/reprodutibilidade
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", "link_prediction_resultados.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"seeds": SEEDS, "execucoes_individuais": raw_runs, "resumo": summary_table},
                f, ensure_ascii=False, indent=2,
            )
        print(f"\n[SUCESSO] Resultados detalhados (todas as execuções + resumo) salvos em: {output_path}")
        
    except Exception as e:
        print(f"\n[ERRO GERAL] Falha no pipeline: {e}")

if __name__ == "__main__":
    main()
