import os
from lxml import etree

def carregar_xml(caminho_arquivo):
    """Lê um arquivo XML individual e extrai o texto."""
    try:
        tree = etree.parse(caminho_arquivo)
        root = tree.getroot()
        # Extrai todo o texto ignorando as tags
        texto_completo = etree.tostring(root, encoding='unicode', method='text')
        return " ".join(texto_completo.split()) # Limpa espaços em branco excessivos
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
        return None

if __name__ == "__main__":
    # Caminho absoluto ou relativo correto
    base_path = os.path.dirname(os.path.abspath(__file__))
    pasta_xmls = os.path.join(base_path, "../data/raw/teste")
    
    # Lista todos os arquivos .xml
    arquivos = [f for f in os.listdir(pasta_xmls) if f.endswith('.xml')]
    
    if not arquivos:
        print(f"Nenhum arquivo XML encontrado em: {os.path.abspath(pasta_xmls)}")
    else:
        print(f"Encontrados {len(arquivos)} arquivos. Iniciando extração...\n")
        
        # O LOOP: Processa um por um
        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(pasta_xmls, nome_arquivo)
            conteudo = carregar_xml(caminho_completo)
            
            if conteudo:
                print(f"--- Arquivo: {nome_arquivo} ---")
                print(f"Prévia: {conteudo[:200]}...") # Mostra só o começo
                print("-" * 30)