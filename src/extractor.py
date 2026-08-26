import os
import xml.etree.ElementTree as ET

class ClinicalTextExtractor:
    def __init__(self, xml_files_list):
        self.xml_files = xml_files_list

    @staticmethod
    def _concept_key(text, tag_type):
        """
        Gera a chave canônica de um conceito clínico: texto normalizado
        (espaços colapsados, maiúsculas) combinado com a categoria semântica
        (tag_type). A categoria entra na chave para evitar unificar termos
        homônimos anotados sob categorias diferentes (ex.: um mesmo texto
        anotado como Sign em um prontuário e como Disease em outro).
        """
        normalized_text = " ".join((text or "").strip().upper().split())
        return f"{normalized_text}|{tag_type}"

    def extract_tags(self):
        """
        Busca profunda por tags de anotação usando XPath (.//).

        Cada ocorrência de um conceito no corpus é agregada em um único nó,
        identificado pela chave canônica (texto normalizado + categoria
        semântica), em vez de um nó novo por ocorrência/prontuário. O campo
        'ocorrencias' registra quantas vezes o conceito foi mencionado no
        corpus inteiro.
        """
        concepts = {}

        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # O './/annotation' faz uma busca profunda em toda a árvore XML
                for anno in root.findall('.//annotation'):
                    text = anno.get('text')
                    tag_type = anno.get('tag')
                    key = self._concept_key(text, tag_type)

                    if key not in concepts:
                        concepts[key] = {
                            "id": key,
                            "text": text,
                            "tag_type": tag_type,
                            "abbr": anno.get('abbr', ''),
                            "ocorrencias": 0
                        }
                    concepts[key]["ocorrencias"] += 1
            except ET.ParseError as e:
                print(f"[ERRO NO PARSER] Falha ao ler {file_path}: {e}")

        return list(concepts.values())

    def extract_relations(self):
        """
        Busca profunda por arcos de relacionamento usando XPath (.//).

        As arestas passam a conectar conceitos únicos (pela chave canônica),
        em vez de ocorrências isoladas por prontuário. Quando a mesma dupla
        de conceitos aparece relacionada (com o mesmo tipo de relação) em
        mais de uma ocorrência no corpus, as instâncias são agregadas em uma
        única aresta, com o campo 'ocorrencias' contando quantas vezes essa
        relação específica foi observada.
        """
        edge_counts = {}

        for file_path in self.xml_files:
            if not os.path.exists(file_path):
                continue

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Mapa local (válido só dentro deste arquivo): id da anotação
                # -> chave canônica do conceito. As relações do XML referenciam
                # ids de anotação locais ao prontuário, não os conceitos únicos.
                local_id_to_key = {}
                for anno in root.findall('.//annotation'):
                    local_id_to_key[anno.get('id')] = self._concept_key(anno.get('text'), anno.get('tag'))

                # O './/rel' garante a captura profunda das arestas
                for rel in root.findall('.//rel'):
                    source_key = local_id_to_key.get(rel.get('annotation1'))
                    target_key = local_id_to_key.get(rel.get('annotation2'))
                    rel_type = rel.get('reltype')

                    if not source_key or not target_key:
                        # A relação referencia uma anotação inexistente neste
                        # arquivo; ignoramos em vez de propagar um id inválido.
                        continue

                    edge_key = (source_key, target_key, rel_type)
                    edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            except ET.ParseError:
                continue

        return [
            {"source": source, "target": target, "rel_type": rel_type, "ocorrencias": count}
            for (source, target, rel_type), count in edge_counts.items()
        ]
