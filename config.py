import os

# Caminho absoluto ou relativo para a pasta com seus 998 arquivos XML
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Configurações de Conexão com o Docker Neo4j (Usando IPv4 para evitar erros)
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "joel2004" # Altere aqui se a sua senha for diferente!