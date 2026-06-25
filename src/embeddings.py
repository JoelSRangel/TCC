import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

class TransEEmbeddingManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_transe_python(self, embedding_dim=64):
        """
        Extrai as triplas (head, relation, tail) do Neo4j e treina 
        o modelo TransE usando a biblioteca PyKeen baseada em PyTorch.
        """
        print("[INFO] Coletando triplas estruturadas (h, r, t) do Neo4j...")
        
        query = """
        MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
        RETURN h.text AS head, r.tipo_num AS relation, t.text AS tail
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            
        if df.empty:
            raise ValueError("[ERRO] Nenhuma tripla encontrada no banco de dados.")
            
        print(f"[INFO] Total de {len(df)} triplas clínicas recuperadas para o TransE.")
        
        # Converte as colunas para o formato string exigido pela PyKeen
        df['head'] = df['head'].astype(str)
        df['relation'] = df['relation'].astype(str)
        df['tail'] = df['tail'].astype(str)
        
        # Cria a fábrica de triplas para o PyTorch
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df[['head', 'relation', 'tail']].values
        )
        
        print(f"[INFO] Treinando o modelo TransE via PyTorch (Dimensão: {embedding_dim})...")
        # Executa o pipeline clássico do TransE
        result_pipeline = pipeline(
            training=triples_factory,
            testing=triples_factory,  # Para simplificação, usamos o mesmo conjunto para teste
            model='TransE',
            model_kwargs=dict(embedding_dim=embedding_dim),
            training_kwargs=dict(num_epochs=10, batch_size=256),
            device='cpu' # Pode mudar para 'cuda' se tiver GPU Nvidia ativa
        )
        
        print("[INFO] Extraindo os vetores gerados pelo modelo...")
        model = result_pipeline.model
        
        # Coleta os mapeamentos de Entidade -> ID
        entity_to_id = triples_factory.entity_to_id
        
        # API ATUALIZADA: Puxa os tensores numéricos usando o resolvedor oficial da PyKeen
        entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
        
        # Busca no Neo4j as tags correspondentes para manter a compatibilidade com o visualizador
        with self.driver.session() as session:
            tag_result = session.run("MATCH (n:Entidade) RETURN n.text AS text, n.tag_type AS tag")
            tag_dict = {record["text"]: record["tag"] for record in tag_result}

        # Monta a lista final de dicionários com a chave 'embedding' para o analisador de cosseno
        embeddings_list = []
        for text, idx in entity_to_id.items():
            embeddings_list.append({
                "text": text,
                "tag": tag_dict.get(text, "Unspecified"),
                "embedding": entity_embeddings[idx].tolist() # Converte o array numpy para lista pura Python
            })
            
        return embeddings_list