from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.embeddings import TransEEmbeddingManager
from src.analysis import EmbeddingAnalyzer
from src.visualizer import EmbeddingVisualizer

def main():
    print("=== PIPELINE FINAL: EXPERIMENTO TRANSE COMPLETO ===")
    
    # 1. Extração de embeddings via PyKeen
    embed_manager = TransEEmbeddingManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    vectors = embed_manager.run_transe_python(embedding_dim=64)
    embed_manager.close()
    
    # 2. Análise Semântica de Cosseno com os Termos Reais do Grafo
    print("\n=== 2. PROCESSAMENTO ANALÍTICO E BUSCA DE COSSENO (TOP 5 REAL) ===")
    analyzer = EmbeddingAnalyzer(vectors)
    
    # Atualizamos os alvos para a realidade da base global
    novos_alvos = ["SEM", "NEGA", "PA", "EDEMA"]
    
    for alvo in novos_alvos:
        nome, similares = analyzer.find_most_similar_by_text(alvo, top_n=3)
        if similares:
            print(f"\nResultados de proximidade para o termo alvo '{nome}':")
            for i, s in enumerate(similares, 1):
                print(f"  {i}º: {s['text']} | Categoria: {s['tag']} | Cosseno: {s['similarity']:.4f}")

    # 3. Plotagem Gráfica com os Termos Identificados pela Centralidade
    print("\n=== 3. PLOTAGEM DO MAPA DE CLUSTERS GEOMÉTRICOS ===")
    visualizer = EmbeddingVisualizer(vectors)
    
    # Destacamos os termos que o próprio grafo provou serem os mais importantes
    visualizer.generate_scatter_plot(target_terms=["SEM", "NEGA", "PA", "EDEMA", "ausente"], output_filename="clusters_transe.png")
    
    print("\n=== [SUCESSO] Branch feature/transe concluída com maestria! ===")

if __name__ == "__main__":
    main()