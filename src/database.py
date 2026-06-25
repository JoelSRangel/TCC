from neo4j import GraphDatabase

class Neo4jInjestor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_clinical_data(self, nodes, edges, chunk_size=5000):
        with self.driver.session() as session:
            print(f"[INFO] Injetando {len(nodes)} nós em lotes de {chunk_size}...")
            for i in range(0, len(nodes), chunk_size):
                chunk = nodes[i:i + chunk_size]
                node_query = """
                UNWIND $batch AS row
                MERGE (n:Entidade {id: row.id})
                ON CREATE SET n.text = row.text, n.tag_type = row.tag_type, n.abbr = row.abbr
                """
                session.run(node_query, batch=chunk)
            
            print(f"[INFO] Injetando {len(edges)} arestas em lotes de {chunk_size}...")
            for i in range(0, len(edges), chunk_size):
                chunk = edges[i:i + chunk_size]
                edge_query = """
                UNWIND $batch AS row
                MATCH (source:Entidade {id: row.source})
                MATCH (target:Entidade {id: row.target})
                MERGE (source)-[r:CONECTA_A]->(target)
                SET r.tipo_num = row.rel_type // <-- Mudamos para tipo_num
                """
                session.run(edge_query, batch=chunk)