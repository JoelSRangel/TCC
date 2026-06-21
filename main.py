import os
import glob
from config import DATA_RAW_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.extractor import ClinicalTextExtractor
from src.database import Neo4jInjestor
from src.embeddings import GraphEmbeddingManager
from src.analysis import EmbeddingAnalyzer
from src.visualizer import EmbeddingVisualizer

def main():
    # 1. Varredura dinâmica de múltiplos arquivos XML usando glob
    print("=== 1. VARREDURA E EXTRAÇÃO EM LOTE ===")
    search_pattern = os.path.join(DATA_RAW_DIR, "*.xml")
    xml_files = glob.glob(search_pattern)
    
    if not xml_files:
        print(f"[ERRO] Nenhum arquivo XML encontrado na pasta: {DATA_RAW_DIR}")
        return
        
    print(f"[INFO] Encontrados {len(xml_files)} arquivos para processamento.")
    for f in xml_files:
        print(f"  -> Carregando: {os.path.basename(f)}")

    # Instancia o extrator refatorado passando a lista completa de arquivos
    extractor = ClinicalTextExtractor(xml_files)
    nodes = extractor.extract_tags()
    edges = extractor.extract_relations()
    print(f"\n[SUCESSO] Extração global concluída: {len(nodes)} nós e {len(edges)} arestas acumulados.")
    
    # 2. Ingestão segura no Neo4j (o mecanismo idempotente absorve o lote sem duplicar)
    print("\n=== 2. INGESTÃO DO GRAFO EXPANDIDO ===")
    try:
        db = Neo4jInjestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        db.insert_clinical_data(nodes, edges)
        db.close()
        print("[SUCESSO] Dados globais persistidos no Neo4j.")
    except Exception as e:
        print(f"[ERRO NO BANCO] {e}")
        return

    # 3. Treinamento do modelo matemático sobre a nova densidade da rede
    print("\n=== 3. GERAÇÃO DE GRAPH EMBEDDINGS (GRAFO EXPANDIDO) ===")
    try:
        embed_manager = GraphEmbeddingManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        vectors = embed_manager.run_pipeline(embedding_dim=64)
        embed_manager.close()
        print(f"[SUCESSO] Total de vetores gerados a partir do lote: {len(vectors)}")
    except Exception as e:
        print(f"[ERRO NOS EMBEDDINGS] {e}")
        return

    # 4. Análise de Similaridade Semântica e Geração do Novo Gráfico
    print("\n=== 4. PROCESSAMENTO ANALÍTICO E VISUAL ===")
    analyzer = EmbeddingAnalyzer(vectors)
    
    # Executa buscas de validação rápida no console para verificar os novos cossenos
    _, _ = analyzer.find_most_similar_by_text("OSTEOMIELITE", top_n=3)
    _, _ = analyzer.find_most_similar_by_text("POI", top_n=3)

    print("\n=== 5. PLOTAGEM DO NOVO MAPA DE CLUSTERS ===")
    visualizer = EmbeddingVisualizer(vectors)
    
    # Mantemos os mesmos termos de interesse destacados, mas agora mapeados no novo espaço multidimensional
    termos_destaque = ["OSTEOMIELITE", "NÃO ESPECIFICADA", "POI", "LAVAGEM", "CURETA"]
    visualizer.generate_scatter_plot(target_terms=termos_destaque, output_filename="clusters_globais_semclinbr.png")

if __name__ == "__main__":
    main()