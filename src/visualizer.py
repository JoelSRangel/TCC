import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

class EmbeddingVisualizer:
    def __init__(self, embeddings_list):
        self.texts = [item['text'] for item in embeddings_list]
        self.tags = [item['tag'] for item in embeddings_list]
        
        raw_matrix = np.array([item['embedding'] for item in embeddings_list])
        if np.iscomplexobj(raw_matrix):
            self.vector_matrix = np.abs(raw_matrix)
        else:
            self.vector_matrix = raw_matrix

    def generate_scatter_plot(self, target_terms=None, output_filename="clusters_complex.png"):
        print("[INFO] Reduzindo dimensões via PCA (64D Complex -> 2D)...")
        pca = PCA(n_components=2, random_state=42)
        vectors_2d = pca.fit_transform(self.vector_matrix)

        plt.figure(figsize=(15, 9))
        unique_tags = list(set(self.tags))
        colors = plt.cm.get_cmap('tab20', len(unique_tags))
        
        for i, tag in enumerate(unique_tags):
            indices = [j for j, t in enumerate(self.tags) if t == tag]
            plt.scatter(vectors_2d[indices, 0], vectors_2d[indices, 1], alpha=0.4, label=tag, s=30, color=colors(i))

        if target_terms:
            target_lower = [t.lower() for t in target_terms]
            for j, text in enumerate(self.texts):
                cleaned_text = text.lower().replace('"', '')
                if cleaned_text in target_lower:
                    plt.annotate(text, (vectors_2d[j, 0], vectors_2d[j, 1]), 
                                 textcoords="offset points", xytext=(5,5), 
                                 ha='center', weight='bold', fontsize=10, 
                                 bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.8)) # Mudamos para laranja no ComplEx
                    plt.scatter(vectors_2d[j, 0], vectors_2d[j, 1], color='red', s=150, edgecolors='black', zorder=5)

        plt.title("Projeção do Espaço Latente ComplEx (Lote Completo SemClinBr)", fontsize=14, pad=15)
        plt.legend(title="Categorias Semânticas", bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 7.5})
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.subplots_adjust(right=0.72, left=0.08, top=0.92, bottom=0.08)

        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[SUCESSO] Gráfico salvo em: {output_path}")