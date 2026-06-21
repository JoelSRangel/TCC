from neo4j import GraphDatabase

class Neo4jInjestor:
    def __init__(self, uri, user, password):
        """
        Inicializa o driver de conexão com o Neo4j.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """
        Fecha a conexão com o banco de dados.
        """
        self.driver.close()

    def insert_clinical_data(self, nodes, edges):
        """
        Orquestra a inserção de nós e arestas dentro de uma sessão no banco.
        """
        with self.driver.session() as session:
            # 1. Ingestão dos Nós
            print("[INFO] Injetando nós no Neo4j...")
            session.execute_write(self._create_nodes_tx, nodes)
            
            # 2. Ingestão das Arestas
            print("[INFO] Criando relacionamentos no Neo4j...")
            session.execute_write(self._create_edges_tx, edges)

    @staticmethod
    def _create_nodes_tx(tx, nodes):
        """
        Query Cypher para criar os nós. Usa o MERGE para evitar duplicatas caso o script rode duas vezes.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (e:Entidade {id: node.id})
        ON CREATE SET e.text = node.text,
                      e.tag_type = node.tag_type,
                      e.abbr = node.abbr
        """
        tx.run(query, nodes=nodes)

    @staticmethod
    def _create_edges_tx(tx, edges):
        """
        Query Cypher para conectar as entidades baseado nos IDs de origem e destino.
        """
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entidade {id: edge.source_id})
        MATCH (target:Entidade {id: edge.target_id})
        
        // Cria o relacionamento de forma dinâmica usando o tipo vindo do XML
        CALL apoc.create.relationship(source, tounicode(upper(edge.rel_type)), {}, target) YIELD rel
        RETURN count(rel)
        """
        # Nota: Como o tipo da aresta muda dinamicamente (negation_of, associated_with), 
        # usamos a função do APOC ou uma query adaptada para criar o tipo dinamicamente.
        # Para simplificar sem depender de plugins externos complexos se a sua imagem community for enxuta,
        # vamos usar uma abordagem nativa segura:
        
        query_native = """
        UNWIND $edges AS edge
        MATCH (source:Entidade {id: edge.source_id})
        MATCH (target:Entidade {id: edge.target_id})
        MERGE (source)-[r:CONECTA_A {tipo: edge.rel_type}]->(target)
        """
        tx.run(query_native, edges=edges)