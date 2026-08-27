import os
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env na raiz do projeto (não versionado — veja .env.example)
load_dotenv()

# Caminhos de Arquivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Credenciais do Neo4j — lidas do .env em vez de hardcoded, para não vazar
# senha em texto puro no repositório (que é público no GitHub).
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_PASSWORD não definida. Copie .env.example para .env "
        "na raiz do projeto e preencha a senha do seu Neo4j."
    )
