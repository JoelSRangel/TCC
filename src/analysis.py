import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingAnalyzer:
    def __init__(self, embeddings_list):
        self.embeddings = embeddings_list
        self.texts = [item['text'] for item in embeddings_list]
        self.tags = [item['tag'] for item in embeddings_list]
        
        raw_matrix = np.array([item['embedding'] for item in embeddings_list])
        
        # Filtro de segurança para extrair o módulo real se a matriz for complexa
        if np.iscomplexobj(raw_matrix):
            self.vector_matrix = np.abs(raw_matrix)
        else:
            self.vector_matrix = raw_matrix

    def find_most_similar_by_text(self, target_text, top_n=3):
        try:
            target_idx = [t.lower().replace('"', '') for t in self.texts].index(target_text.lower())
        except ValueError:
            print(f"[AVISO] Termo '{target_text}' não localizado no vocabulário espacial.")
            return None, []

        target_vector = self.vector_matrix[target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, self.vector_matrix).flatten()
        most_similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in most_similar_indices:
            if idx == target_idx:
                continue
            results.append({
                "text": self.texts[idx],
                "tag": self.tags[idx],
                "similarity": float(similarities[idx])
            })
            if len(results) == top_n:
                break
        return self.texts[target_idx], results