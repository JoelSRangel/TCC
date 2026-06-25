from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.embeddings import RotateEmbeddingManager
from src.analysis import EmbeddingAnalyzer
from src.visualizer import EmbeddingVisualizer

def main():
    print("=== PIPELINE FINAL: EXPERIMENTO ROTATE COMPLETO ===")
    
    # 1. Execução do modelo RotatE
    manager = RotateEmbeddingManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    vectors = manager.run_rotate_python(embedding_dim=64)
    manager.close()
    
    # 2. Processamento Analítico de Cosseno (Top 5 Real)
    print("\n=== 2. PROCESSAMENTO ANALÍTICO E BUSCA DE COSSENO (TOP 5 REAL) ===")
    analyzer = EmbeddingAnalyzer(vectors)
    novos_alvos = ["SEM", "NEGA", "PA", "EDEMA"]
    
    for alvo in novos_alvos:
        nome, similares = analyzer.find_most_similar_by_text(alvo, top_n=3)
        if similares:
            print(f"\nResultados de proximidade RotatE para o termo alvo '{nome}':")
            for i, s in enumerate(similares, 1):
                print(f"  {i}º: {s['text']} | Categoria: {s['tag']} | Cosseno: {s['similarity']:.4f}")

    # 3. Plotagem Gráfica Final
    print("\n=== 3. PLOTAGEM DO MAPA DE CLUSTERS GEOMÉTRICOS ROTACIONAIS ===")
    visualizer = EmbeddingVisualizer(vectors)
    visualizer.generate_scatter_plot(target_terms=["SEM", "NEGA", "PA", "EDEMA", "ausente"], output_filename="clusters_rotate.png")
    
    print("\n=== [SUCESSO] Branch feature/rotate concluída com maestria! ===")

if __name__ == "__main__":
    main()