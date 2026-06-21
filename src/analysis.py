import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingAnalyzer:
    def __init__(self, embeddings_list):
        """
        Inicializa o analisador estruturando os dados em matrizes numpy para performance.
        """
        self.embeddings = embeddings_list
        self.ids = [item['id'] for item in embeddings_list]
        self.texts = [item['text'] for item in embeddings_list]
        self.tags = [item['tag'] for item in embeddings_list]
        
        # Converte a lista de listas de float em uma matriz bidimensional do NumPy
        self.vector_matrix = np.array([item['vector'] for item in embeddings_list])

    def find_most_similar_by_text(self, target_text, top_n=5):
        """
        Busca uma entidade pelo texto literal e calcula os N vizinhos mais proximos
        usando a Similaridade de Cosseno.
        """
        # Encontra o indice do termo alvo (ignora maiusculas/minusculas para evitar erros)
        try:
            target_idx = [t.lower() for t in self.texts].index(target_text.lower())
        except ValueError:
            print(f"[AVISO] Termo '{target_text}' nao encontrado no grafo atual.")
            return []

        target_vector = self.vector_matrix[target_idx].reshape(1, -1)
        
        # Calcula o cosseno entre o vetor alvo e TODOS os outros vetores da matriz
        similarities = cosine_similarity(target_vector, self.vector_matrix).flatten()
        
        # Ordena os indices dos maiores valores para os menores
        most_similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in most_similar_indices:
            # Ignora o proprio termo alvo na listagem de similares
            if idx == target_idx:
                continue
                
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "tag": self.tags[idx],
                "similarity": float(similarities[idx])
            })
            
            if len(results) == top_n:
                break
                
        return self.texts[target_idx], results