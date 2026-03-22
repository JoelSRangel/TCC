import os
import json
import ollama
from lxml import etree
from neo4j import GraphDatabase
import re
import unicodedata

# Configurações do Neo4j (Use a senha que você definiu no Docker)
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "joel2004" 

class GraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    def close(self):
        self.driver.close()

    def add_triple(self, s, r, o):
        with self.driver.session() as session:
            # Cypher query: Cria nós e a relação (MERGE evita duplicatas)
            query = (
                "MERGE (a:Entidade {name: $s}) "
                "MERGE (b:Entidade {name: $o}) "
                f"MERGE (a)-[rel:{r}]->(b)"
            )
            session.run(query, s=s, o=o)
            
def sanitizar_relacao(texto):
    """Remove acentos e caracteres especiais para nomes de relações no Neo4j."""
    # 1. Remove acentos (Transforma 'Â' em 'A')
    texto = "".join(c for c in unicodedata.normalize('NFD', texto) 
                    if unicodedata.category(c) != 'Mn')
    
    # 2. Substitui qualquer coisa que não seja letra ou número por '_'
    texto = re.sub(r'[^a-zA-Z0-9]', '_', texto)
    
    # 3. Remove múltiplos '_' seguidos e limpa as bordas
    texto = re.sub(r'_+', '_', texto).strip('_')
    
    return texto.upper()

def extrair_texto_xml(caminho):
    tree = etree.parse(caminho)
    return " ".join(etree.tostring(tree.getroot(), encoding='unicode', method='text').split())

def processar_com_llama(texto):
    # Melhoramos o prompt para ser mais rígido
    prompt = f"""
    Analise o texto médico e extraia entidades e relações.
    Responda EXCLUSIVAMENTE um JSON estruturado:
    {{"triplas": [{{"s": "sujeito", "r": "relacao", "o": "objeto"}}]}}
    
    Texto: {texto[:1200]}
    """
    
    response = ollama.chat(model='llama3', 
                           messages=[{'role': 'user', 'content': prompt}],
                           format='json')
    
    try:
        conteudo = json.loads(response['message']['content'])
        # Normalização: garantir que as chaves existam mesmo que a IA erre o nome
        triplas_limpas = []
        # Tenta encontrar a lista de triplas independente do nome da chave
        lista_crua = next(iter(conteudo.values())) if isinstance(conteudo, dict) else []
        
        for t in lista_crua:
            # Pega os valores por posição ou chave, prevenindo o erro 'r'
            s = t.get('s') or t.get('sujeito') or t.get('subject')
            r = t.get('r') or t.get('relacao') or t.get('relation')
            o = t.get('o') or t.get('objeto') or t.get('object')
            
            if s and r and o:
                triplas_limpas.append({'s': s, 'r': r, 'o': o})
        
        return triplas_limpas
    except json.JSONDecodeError as e:
        print(f" -> Erro de sintaxe JSON da IA. Tentando recuperar...")
        return []

if __name__ == "__main__":
    db = GraphManager()
    pasta = "./data/raw/teste"
    
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".xml"):
            print(f"Processando {arquivo}...")
            texto = extrair_texto_xml(os.path.join(pasta, arquivo))
            
            try:
                triplas = processar_com_llama(texto)
                for t in triplas:
                    # Use a nova função aqui
                    rel_limpa = sanitizar_relacao(str(t['r']))
                    
                    # Se a relação ficou vazia após a limpeza, use um padrão
                    if not rel_limpa:
                        rel_limpa = "RELACIONADO_A"
                        
                    db.add_triple(str(t['s']), rel_limpa, str(t['o']))
            except Exception as e:
                print(f"Erro no arquivo {arquivo}: {e}")

    db.close()
    print("Ingestão concluída!")