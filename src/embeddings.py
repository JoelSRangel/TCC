import pandas as pd
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

class RotateEmbeddingManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_rotate_python(self, embedding_dim=64):
        print("[INFO] Coletando triplas válidas para o RotatE...")
        query = """
        MATCH (h:Entidade)-[r:CONECTA_A]->(t:Entidade)
        WHERE r.tipo_num IS NOT NULL
        RETURN h.text AS head, r.tipo_num AS relation, t.text AS tail
        """
        with self.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            
        if df.empty:
            raise ValueError("[ERRO] Nenhuma tripla válida com 'tipo_num' foi encontrada no banco.")
            
        df['head'] = df['head'].astype(str)
        df['relation'] = df['relation'].astype(str)
        df['tail'] = df['tail'].astype(str)
            
        print(f"[SUCESSO] {len(df)} triplas clínicas prontas.")
        
        # 1. Criação da fábrica de triplas PyKeen
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df[['head', 'relation', 'tail']].values
        )
        
        print(f"[INFO] Iniciando treinamento do modelo RotatE via PyTorch (Dimensão: {embedding_dim})...")
        # 2. Executa o pipeline do RotatE
        result_pipeline = pipeline(
            training=triples_factory,
            testing=triples_factory,
            model='RotatE',
            model_kwargs=dict(embedding_dim=embedding_dim),
            training_kwargs=dict(num_epochs=10, batch_size=256),
            device='cpu'
        )
        
        print("[INFO] Extraindo os vetores complexos gerados pela rotação...")
        model = result_pipeline.model
        entity_to_id = triples_factory.entity_to_id
        
        # Coleta as representações da API do RotatE
        entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
        
        # Busca as tags no Neo4j para mapeamento do visualizador
        with self.driver.session() as session:
            tag_result = session.run("MATCH (n:Entidade) RETURN n.text AS text, n.tag_type AS tag")
            tag_dict = {record["text"]: record["tag"] for record in tag_result}

        # Monta a lista padrão compatível com os analisadores
        embeddings_list = []
        for text, idx in entity_to_id.items():
            embeddings_list.append({
                "text": text,
                "tag": tag_dict.get(text, "Unspecified"),
                "embedding": entity_embeddings[idx].tolist()
            })
            
        return embeddings_list