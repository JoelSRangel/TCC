from neo4j import GraphDatabase
import numpy as np


class GraphStatsAnalyzer:
    """
    Calcula estatísticas estruturais do grafo de conhecimento clínico:
    componentes conexos, nós isolados e distribuição de grau.

    Todas as métricas tratam o grafo como não-direcionado (mesma orientação
    usada na projeção do Node2Vec em embeddings.py), para que os números
    sejam diretamente comparáveis com o restante do trabalho.
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph_name = "clinical_projection"

    def close(self):
        self.driver.close()

    def get_isolated_nodes(self):
        """
        Retorna os nós sem nenhuma relação (id, text, tag_type), para que
        outros scripts (main.py, benchmarks de clustering) possam excluí-los
        de avaliações de qualidade de embedding (PCA, similaridade,
        clustering) — nós sem aresta não têm sinal estrutural aprendido pelo
        Node2Vec, então incluí-los distorce essas métricas. Abre e fecha sua
        própria sessão, então pode ser chamado de forma independente de
        compute_stats().
        """
        query = """
        MATCH (n:Entidade)
        WHERE NOT (n)--()
        RETURN n.id AS id, n.text AS text, n.tag_type AS tag_type
        """
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query)]

    def compute_stats(self):
        with self.driver.session() as session:
            self._ensure_projection(session)

            stats = {}
            stats["total_nos"] = self._count_nodes(session)
            stats["total_arestas"] = self._count_relationships(session)
            stats["nos_isolados"] = self._count_isolated_nodes(session)
            stats.update(self._weakly_connected_components(session))
            stats.update(self._degree_distribution(session))
            return stats

    def _ensure_projection(self, session):
        """
        Reaproveita a projeção 'clinical_projection' se ela já estiver em
        memória (por exemplo, logo após rodar main.py); caso contrário, cria
        uma nova com a mesma orientação (UNDIRECTED) usada para os embeddings.
        """
        existing = [
            record["graphName"]
            for record in session.run("CALL gds.graph.list() YIELD graphName RETURN graphName")
        ]

        if self.graph_name in existing:
            print(f"[INFO] Reaproveitando projeção '{self.graph_name}' já em memória.")
            return

        print(f"[INFO] Projeção '{self.graph_name}' não encontrada. Criando...")
        query = """
        CALL gds.graph.project(
          $graph_name,
          'Entidade',
          {
            CONECTA_A: {
              type: 'CONECTA_A',
              orientation: 'UNDIRECTED'
            }
          }
        )
        """
        session.run(query, graph_name=self.graph_name)

    def _count_nodes(self, session):
        return session.run("MATCH (n:Entidade) RETURN count(n) AS total").single()["total"]

    def _count_relationships(self, session):
        return session.run("MATCH ()-[r:CONECTA_A]->() RETURN count(r) AS total").single()["total"]

    def _count_isolated_nodes(self, session):
        """
        Nós sem nenhuma relação — calculado direto em Cypher (independe da
        projeção do GDS), contando conceitos que apareceram no corpus mas
        nunca foram relacionados a outro conceito.
        """
        query = "MATCH (n:Entidade) WHERE NOT (n)--() RETURN count(n) AS total"
        return session.run(query).single()["total"]

    def _weakly_connected_components(self, session):
        """
        Componentes conexos (Weakly Connected Components). Como a projeção já
        é UNDIRECTED, WCC aqui equivale a componentes conexos "normais" de
        teoria dos grafos.
        """
        query = """
        CALL gds.wcc.stream($graph_name)
        YIELD nodeId, componentId
        RETURN componentId, count(*) AS tamanho
        ORDER BY tamanho DESC
        """
        result = session.run(query, graph_name=self.graph_name)
        tamanhos = [record["tamanho"] for record in result]

        if not tamanhos:
            return {"num_componentes_conexos": 0, "tamanho_maior_componente": 0, "pct_nos_no_maior_componente": 0.0}

        total_nos = sum(tamanhos)
        return {
            "num_componentes_conexos": len(tamanhos),
            "tamanho_maior_componente": tamanhos[0],
            "pct_nos_no_maior_componente": round(100 * tamanhos[0] / total_nos, 2),
        }

    def _degree_distribution(self, session):
        """
        Grau de cada nó via GDS Degree Centrality. Como a projeção já é
        UNDIRECTED, NÃO especificamos orientation de novo aqui — o grau
        resultante já é o grau não-direcionado correto (cada aresta conta 1
        para cada um dos dois nós que ela conecta, sem dobrar).
        """
        query = """
        CALL gds.degree.stream($graph_name)
        YIELD nodeId, score
        RETURN score AS grau
        """
        result = session.run(query, graph_name=self.graph_name)
        graus = np.array([record["grau"] for record in result], dtype=float)

        if graus.size == 0:
            return {"grau_medio": 0.0, "grau_mediano": 0.0, "grau_min": 0.0, "grau_max": 0.0, "grau_desvio_padrao": 0.0}

        return {
            "grau_medio": round(float(np.mean(graus)), 3),
            "grau_mediano": float(np.median(graus)),
            "grau_min": float(np.min(graus)),
            "grau_max": float(np.max(graus)),
            "grau_desvio_padrao": round(float(np.std(graus)), 3),
        }
