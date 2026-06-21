from neo4j import GraphDatabase

class GraphEmbeddingManager:
    def __init__(self, uri, user, password):
        """
        Inicializa o driver para gerenciar os algoritmos do GDS.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph_name = "clinical_projection"

    def close(self):
        self.driver.close()

    def run_pipeline(self, embedding_dim=64):
        """
        Orquestra a projeção do grafo, execução do Node2Vec e coleta dos resultados.
        """
        with self.driver.session() as session:
            # 1. Limpa projeções antigas se existirem para evitar conflitos
            self._drop_projection_if_exists(session)
            
            # 2. Cria a nova projeção na memória RAM do Neo4j
            self._create_graph_projection(session)
            
            # 3. Executa o Node2Vec e recupera os vetores
            print(f"[INFO] Executando Node2Vec (Dimensão: {embedding_dim})...")
            embeddings = self._execute_node2vec(session, embedding_dim)
            
            return embeddings

    def _drop_projection_if_exists(self, session):
        """
        Verifica se a projeção existe antes de tentar deletá-la, 
        evitando exceções em bancos de dados zerados.
        """
        check_query = "CALL gds.graph.list() YIELD graphName RETURN graphName"
        result = session.run(check_query)
        
        # Coleta os nomes de todas as projeções ativas na memória
        existing_graphs = [record["graphName"] for record in result]
        
        if self.graph_name in existing_graphs:
            print(f"[INFO] Projeção '{self.graph_name}' encontrada. Removendo da memória...")
            session.run("CALL gds.graph.drop($graph_name) YIELD graphName", graph_name=self.graph_name)
            print("[INFO] Projeção antiga removida com sucesso.")
        else:
            print("[INFO] Nenhuma projeção anterior encontrada. Pronto para criar uma nova.")

    def _create_graph_projection(self, session):
        """
        Projeta o grafo 'Entidade' e as relações 'CONECTA_A' para a memória do GDS.
        """
        print("[INFO] Criando projeção do grafo na memória do GDS...")
        query = """
        CALL gds.graph.project(
          $graph_name,
          'Entidade',
          {
            CONECTA_A: {
              type: 'CONECTA_A',
              orientation: 'UNDIRECTED' // Node2Vec performa melhor em grafos não-direcionados
            }
          }
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        result = session.run(query, graph_name=self.graph_name).single()
        print(f"[SUCESSO] Grafo projetado: {result['nodeCount']} nós e {result['relationshipCount']} arestas carregados na RAM.")

    def _execute_node2vec(self, session, embedding_dim):
        """
        Roda o Node2Vec no modo 'stream' para retornar os vetores diretamente para o Python.
        """
        query = """
        CALL gds.node2vec.stream($graph_name, {
          embeddingDimension: $dim,
          walkLength: 10,
          walksPerNode: 10
        })
        YIELD nodeId, embedding
        MATCH (n:Entidade) WHERE id(n) = nodeId
        RETURN n.id AS entity_id, n.text AS text, n.tag_type AS tag, embedding
        """
        result = session.run(query, graph_name=self.graph_name, dim=embedding_dim)
        
        embeddings_list = []
        for record in result:
            embeddings_list.append({
                "id": record["entity_id"],
                "text": record["text"],
                "tag": record["tag"],
                "vector": record["embedding"] # Lista de floats (dimensão d)
            })
        return embeddings_list