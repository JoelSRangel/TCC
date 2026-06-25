import os
import xml.etree.ElementTree as ET

class ClinicalTextExtractor:
    def __init__(self, xml_files_list):
        self.xml_files = xml_files_list

    def extract_tags(self):
        all_nodes = []
        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue
            file_name = os.path.basename(file_path).replace(".xml", "")
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                for anno in root.findall('.//annotation'):
                    all_nodes.append({
                        "id": f"{file_name}_{anno.get('id')}",
                        "tag_type": anno.get('tag'),
                        "text": anno.get('text'),
                        "abbr": anno.get('abbr', '')
                    })
            except ET.ParseError:
                continue
        return all_nodes

    def extract_relations(self):
        all_edges = []
        
        # Dicionario de mapeamento para transformar texto em numero (requisito do GDS KGE)
        rel_mapping = {
            "associated_with": 1,
            "negation_of": 2
        }
        
        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue
                
            file_name = os.path.basename(file_path).replace(".xml", "")
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                for rel in root.findall('.//rel'):
                    original_type = rel.get('reltype')
                    
                    # Converte para o numero correspondente. Se for um tipo raro, padroniza como 3
                    numeric_type = rel_mapping.get(original_type, 3)
                    
                    all_edges.append({
                        "source": f"{file_name}_{rel.get('annotation1')}",
                        "target": f"{file_name}_{rel.get('annotation2')}",
                        "rel_type": numeric_type # Agora enviamos um INTEIRO
                    })
            except ET.ParseError:
                continue
        return all_edges