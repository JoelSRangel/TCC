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
            # 1. Ingestão dos Nós (conceitos únicos)
            print("[INFO] Injetando nós (conceitos únicos) no Neo4j...")
            session.execute_write(self._create_nodes_tx, nodes)

            # 2. Ingestão das Arestas
            print("[INFO] Criando relacionamentos no Neo4j...")
            session.execute_write(self._create_edges_tx, edges)

    @staticmethod
    def _create_nodes_tx(tx, nodes):
        """
        Query Cypher para criar os nós. O MERGE usa a chave canônica do
        conceito (node.id = texto normalizado + categoria semântica,
        calculada em extractor.py), unificando ocorrências do mesmo conceito
        em um único nó em vez de criar um nó por ocorrência.

        Usamos SET (em vez de ON CREATE SET) de propósito: como
        extractor.py já agrega as ocorrências do corpus inteiro em cada
        execução, reexecutar o pipeline deve sempre atualizar as
        propriedades (inclusive 'ocorrencias') para o valor mais recente,
        em vez de travar no valor da primeira execução.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (e:Entidade {id: node.id})
        SET e.text = node.text,
            e.tag_type = node.tag_type,
            e.abbr = node.abbr,
            e.ocorrencias = node.ocorrencias
        """
        tx.run(query, nodes=nodes)

    @staticmethod
    def _create_edges_tx(tx, edges):
        """
        Query Cypher para conectar os nós de conceito com base nos campos
        'source'/'target' produzidos por extractor.extract_relations().

        Correção: a versão anterior desta query referenciava
        'edge.source_id'/'edge.target_id', que não existem no dicionário
        de aresta (as chaves são 'source'/'target'). Em Cypher, acessar uma
        chave inexistente de um mapa retorna null, então o MATCH nunca
        encontrava os nós e nenhuma relação era criada por este caminho —
        o bloco antigo baseado em apoc.create.relationship, que também
        usava os mesmos nomes de campo incorretos e nunca chegava a ser
        executado (só 'query_native' era chamada), foi removido.
        """
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entidade {id: edge.source})
        MATCH (target:Entidade {id: edge.target})
        MERGE (source)-[r:CONECTA_A {tipo: edge.rel_type}]->(target)
        SET r.ocorrencias = edge.ocorrencias
        """
        tx.run(query, edges=edges)
