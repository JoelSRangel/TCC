import os
import re
import unicodedata
from lxml import etree
from neo4j import GraphDatabase

# Configurações do Neo4j
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "joel2004" # TROQUE PELA SUA SENHA

class TCCGraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    def close(self):
        self.driver.close()

    def sanitizar(self, texto):
        """Limpa nomes para o Neo4j (Remove acentos e espaços)"""
        if not texto: return "SEM_NOME"
        nfkd = unicodedata.normalize('NFKD', texto)
        apenas_ascii = nfkd.encode('ASCII', 'ignore').decode('ASCII')
        return re.sub(r'[^a-zA-Z0-9_]', '_', apenas_ascii).upper().strip('_')

    def inserir_dados(self, tags, relacoes, arquivo_id):
        with self.driver.session() as session:
            # 1. Inserir todos os nós (TAGS)
            for t_id, info in tags.items():
                session.run(
                    "MERGE (e:Entidade {id_global: $id_global}) "
                    "SET e.texto = $texto, e.tipo = $tipo, e.arquivo = $arquivo",
                    id_global=f"{arquivo_id}_{t_id}",
                    texto=info['text'],
                    tipo=info['tag'],
                    arquivo=arquivo_id
                )

            # 2. Inserir todas as conexões (RELATIONS)
            for rel in relacoes:
                # Criamos a relação entre os IDs únicos do arquivo
                rel_type = self.sanitizar(rel['type'])
                session.run(
                    f"MATCH (a:Entidade {{id_global: $id1}}) "
                    f"MATCH (b:Entidade {{id_global: $id2}}) "
                    f"MERGE (a)-[:{rel_type}]->(b)",
                    id1=f"{arquivo_id}_{rel['a1']}",
                    id2=f"{arquivo_id}_{rel['a2']}"
                )

def extrair_estruturado(caminho):
    """Extrai Tags e Relações do XML anotado"""
    tree = etree.parse(caminho)
    root = tree.getroot()
    
    tags = {}
    for node in root.xpath("//annotation"):
        tags[node.get("id")] = {
            "text": node.get("text"),
            "tag": node.get("tag")
        }
        
    relacoes = []
    for rel in root.xpath("//rel"):
        relacoes.append({
            "a1": rel.get("annotation1"),
            "a2": rel.get("annotation2"),
            "type": rel.get("reltype")
        })
        
    return tags, relacoes

if __name__ == "__main__":
    manager = TCCGraphManager()
    pasta = "./data/raw/teste"
    
    for nome_arq in os.listdir(pasta):
        if nome_arq.endswith(".xml"):
            print(f"Processando dados anotados de: {nome_arq}")
            try:
                t, r = extrair_estruturado(os.path.join(pasta, nome_arq))
                manager.inserir_dados(t, r, nome_arq)
            except Exception as e:
                print(f"Erro no arquivo {nome_arq}: {e}")

    manager.close()
    print("\n--- Ingestão Estruturada Concluída! ---")