"""
Item 8 do plano de acao pos-banca: prototipo de RAG (Retrieval-Augmented
Generation) usando o grafo de conhecimento clinico como fonte de contexto.

Combina duas fontes de evidencia para uma pergunta em linguagem natural:
1. Relacoes diretas no grafo (Cypher) a partir das entidades mencionadas
   na pergunta -- evidencia estrutural, explicita no corpus.
2. Vizinhos mais proximos no espaco de embeddings RotatE (similaridade de
   cosseno) -- evidencia semantica/latente, o que justifica o uso de Graph
   Embeddings em veze so uma busca por texto no grafo.

O reconhecimento de qual(is) entidade(s) a pergunta menciona e feito por
correspondencia de substring do campo `text` de cada entidade dentro da
pergunta (normalizado para maiusculas) -- suficiente para um prototipo
leve sobre um vocabulario fixo e conhecido (o corpus SemClinBr), sem
exigir um componente de NER a parte. Limitacao documentada: termos muito
curtos ou genericos podem gerar falsos positivos; por isso ha um tamanho
minimo de texto e uma deduplicacao de sobreposicoes (ver
encontrar_entidades_mencionadas).
"""
import json
import os

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

EMBEDDINGS_PATH = os.path.join("results", "rag_embeddings.npz")
METADATA_PATH = os.path.join("results", "rag_entity_metadata.json")

# Similaridade de cosseno minima para um vizinho semantico entrar no
# contexto. Achado ao testar o prototipo (pergunta sobre "QUEIXA", ver
# plano de acao): os vizinhos mais proximos no espaco RotatE apareciam com
# similaridade de ~0,29-0,35 (escala -1 a 1) -- fraca em termos absolutos,
# mas o LLM tratava esse numero como evidencia forte por vir quantificado,
# e chegava a preferi-lo a uma relacao direta do grafo (evidencia bem mais
# confiavel). Filtrar abaixo deste limiar evita apresentar ruido do espaco
# latente como se fosse sinal -- consistente com o proprio Silhouette baixo
# medido nos itens 4/5 (o espaco nao e fortemente discriminativo).
SIMILARIDADE_MINIMA = 0.5


class GraphRAGRetriever:
    def __init__(self):
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Embeddings/metadados do RAG nao encontrados em '{os.path.dirname(EMBEDDINGS_PATH)}'. "
                "Rode 'python treinar_embeddings_rag.py' antes de usar o prototipo."
            )

        dados = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        self.ids = dados["ids"]
        self.embeddings = dados["embeddings"]
        self.id_to_index = {entity_id: i for i, entity_id in enumerate(self.ids)}

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def encontrar_entidades_mencionadas(self, pergunta, max_entidades=3):
        """
        Para cada entidade do grafo, checa se o texto dela aparece dentro
        da pergunta. Prioriza textos mais longos primeiro (evita que um
        termo curto e generico, contido dentro de um termo mais
        especifico, "roube" o match), e descarta candidatos cujo texto
        esteja contido em outro ja selecionado.
        """
        pergunta_normalizada = pergunta.upper()
        candidatos = []
        for entity_id, meta in self.metadata.items():
            texto = (meta.get("text") or "").upper().strip()
            if texto and len(texto) >= 4 and texto in pergunta_normalizada:
                candidatos.append((entity_id, texto))

        candidatos.sort(key=lambda par: len(par[1]), reverse=True)

        selecionados = []
        cobertos = []
        for entity_id, texto in candidatos:
            if any(texto in outro for outro in cobertos):
                continue
            selecionados.append(entity_id)
            cobertos.append(texto)
            if len(selecionados) >= max_entidades:
                break
        return selecionados

    def vizinhos_diretos(self, entity_id, limite=5):
        """Relacoes explicitas no grafo (1 salto), nos dois sentidos."""
        query = """
        MATCH (e:Entidade {id: $id})-[r:CONECTA_A]-(vizinho:Entidade)
        RETURN vizinho.id AS id, vizinho.text AS text, vizinho.tag_type AS tag_type,
               r.tipo AS relacao, startNode(r) = e AS saindo
        LIMIT $limite
        """
        with self.driver.session() as session:
            registros = list(session.run(query, id=entity_id, limite=limite))
        return [dict(r) for r in registros]

    def vizinhos_semanticos(self, entity_id, top_k=5, similaridade_minima=SIMILARIDADE_MINIMA):
        """
        Vizinhos mais proximos no espaco RotatE (cosseno; os vetores ja
        estao normalizados L2 pelo treinamento, entao o produto interno e
        diretamente a similaridade de cosseno), excluindo a propria
        entidade. Evidencia que a busca puramente estrutural (so arestas
        explicitas) nao captura: conceitos semanticamente parecidos mesmo
        sem relacao direta anotada no corpus.

        Descarta candidatos com similaridade abaixo de
        `similaridade_minima` (ver SIMILARIDADE_MINIMA acima) -- um vizinho
        "mais proximo" ainda pode estar longe em termos absolutos, e nesse
        caso e melhor nao devolver nada do que devolver ruido travestido de
        evidencia quantificada. Pode retornar uma lista vazia (entidade sem
        vizinhos suficientemente proximos no espaco latente).
        """
        if entity_id not in self.id_to_index:
            return []
        idx = self.id_to_index[entity_id]
        vetor = self.embeddings[idx]
        similaridades = self.embeddings @ vetor
        ordem = np.argsort(-similaridades)

        resultados = []
        for i in ordem:
            candidato_id = self.ids[i]
            if candidato_id == entity_id:
                continue
            sim = float(similaridades[i])
            if sim < similaridade_minima:
                # ordem e decrescente por similaridade -- assim que cai
                # abaixo do limiar, todo o resto tambem cai.
                break
            meta = self.metadata.get(candidato_id, {})
            resultados.append({
                "id": candidato_id,
                "text": meta.get("text"),
                "tag_type": meta.get("tag_type"),
                "similaridade": sim,
            })
            if len(resultados) >= top_k:
                break
        return resultados

    def montar_contexto(self, pergunta, max_entidades=3, vizinhos_diretos_por_entidade=5,
                         vizinhos_semanticos_por_entidade=5, similaridade_minima=SIMILARIDADE_MINIMA):
        """
        Orquestra a recuperacao: identifica entidades mencionadas na
        pergunta, busca vizinhos diretos (estrutural) e semanticos
        (embeddings, filtrados por similaridade_minima) para cada uma, e
        monta o texto de contexto em portugues a ser injetado no prompt do
        LLM. Retorna (contexto_texto, detalhes) -- detalhes e usado pela
        interface para exibir o que foi recuperado, separado da resposta
        final do LLM.
        """
        entidades = self.encontrar_entidades_mencionadas(pergunta, max_entidades=max_entidades)

        linhas_contexto = []
        detalhes = {"entidades_reconhecidas": [], "vizinhos_diretos": {}, "vizinhos_semanticos": {}}

        for entity_id in entidades:
            meta = self.metadata.get(entity_id, {})
            texto_entidade = meta.get("text", entity_id)
            detalhes["entidades_reconhecidas"].append({"id": entity_id, **meta})

            diretos = self.vizinhos_diretos(entity_id, limite=vizinhos_diretos_por_entidade)
            detalhes["vizinhos_diretos"][entity_id] = diretos
            for v in diretos:
                if v["saindo"]:
                    linhas_contexto.append(f"{texto_entidade} --[{v['relacao']}]--> {v['text']} ({v['tag_type']})")
                else:
                    linhas_contexto.append(f"{v['text']} ({v['tag_type']}) --[{v['relacao']}]--> {texto_entidade}")

            semanticos = self.vizinhos_semanticos(
                entity_id, top_k=vizinhos_semanticos_por_entidade, similaridade_minima=similaridade_minima
            )
            detalhes["vizinhos_semanticos"][entity_id] = semanticos
            for v in semanticos:
                linhas_contexto.append(
                    f"{texto_entidade} e semanticamente proximo de {v['text']} ({v['tag_type']}) "
                    f"no espaco de embeddings [similaridade={v['similaridade']:.2f}]"
                )

        contexto_texto = "\n".join(linhas_contexto)
        return contexto_texto, detalhes
