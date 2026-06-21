import os
import xml.etree.ElementTree as ET

class ClinicalTextExtractor:
    def __init__(self, xml_files_list):
        self.xml_files = xml_files_list

    def extract_tags(self):
        """
        Busca profunda por tags de anotacao usando XPath (.//)
        """
        all_nodes = []
        
        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue
                
            file_name = os.path.basename(file_path).replace(".xml", "")
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # O './/annotation' faz uma busca profunda em toda a árvore XML
                for anno in root.findall('.//annotation'):
                    original_id = anno.get('id')
                    global_id = f"{file_name}_{original_id}"
                    
                    all_nodes.append({
                        "id": global_id,
                        "tag_type": anno.get('tag'),
                        "text": anno.get('text'),
                        "abbr": anno.get('abbr', '')
                    })
            except ET.ParseError as e:
                print(f"[ERRO NO PARSER] Falha ao ler {file_path}: {e}")
                
        return all_nodes

    def extract_relations(self):
        """
        Busca profunda por arcos de relacionamento usando XPath (.//)
        """
        all_edges = []
        
        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue
                
            file_name = os.path.basename(file_path).replace(".xml", "")
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # O './/rel' garante a captura profunda das arestas
                for rel in root.findall('.//rel'):
                    global_source = f"{file_name}_{rel.get('annotation1')}"
                    global_target = f"{file_name}_{rel.get('annotation2')}"
                    
                    all_edges.append({
                        "source": global_source,
                        "target": global_target,
                        "rel_type": rel.get('reltype')
                    })
            except ET.ParseError:
                continue
                
        return all_edges