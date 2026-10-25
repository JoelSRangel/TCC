"""
Fatos sobre as relações do SemClinBr citados no texto do TCC.

Recalcula, a partir dos XMLs em data/raw, os números usados no cap. 2
(descrição do SemClinBr e dos modelos TransE, ComplEx e RotatE), no cap. 3
(Base de Dados e Modelagem do Grafo) e no cap. 4 (limitações dos modelos KGE).

O grafo unificado usa a mesma chave de src/extractor.py: texto normalizado
(maiúsculas, espaços repetidos removidos) + categoria bruta da anotação.

Uso (da raiz do repositório):
    python analises/01_relacoes.py [pasta_com_os_xml]
"""
import collections
import glob
import os
import sys
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "raw")

# Marcadores que descrevem a menção, e não uma categoria clínica
# (mesma regra de normalize_tag() em cluster_benchmark.py).
MARCADORES = {"Abbreviation", "Negation"}


def chave(texto, tag):
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def categoria(tag_bruta):
    partes = []
    for parte in (tag_bruta or "").split("|"):
        if parte not in partes:
            partes.append(parte)
    substantivas = [p for p in partes if p not in MARCADORES]
    if substantivas:
        return substantivas[0]
    return partes[0] if partes else tag_bruta


def texto_de(conceito):
    return conceito.split("|")[0]


def ler_corpus(pasta):
    """Lê os XMLs e devolve as arestas do grafo unificado com suas ocorrências."""
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.xml")))
    ocorrencias = collections.Counter()   # (origem, destino, tipo) -> nº de vezes no corpus
    tag_do_conceito = {}
    por_tipo_no_xml = collections.Counter()
    for arquivo in arquivos:
        raiz = ET.parse(arquivo).getroot()
        local = {}
        for anotacao in raiz.findall(".//annotation"):
            k = chave(anotacao.get("text"), anotacao.get("tag"))
            local[anotacao.get("id")] = k
            tag_do_conceito[k] = anotacao.get("tag")
        for rel in raiz.findall(".//rel"):
            tipo = rel.get("reltype")
            por_tipo_no_xml[tipo] += 1
            origem = local.get(rel.get("annotation1"))
            destino = local.get(rel.get("annotation2"))
            if origem and destino:
                ocorrencias[(origem, destino, tipo)] += 1
    return len(arquivos), ocorrencias, tag_do_conceito, por_tipo_no_xml


def pct(parte, total):
    return 100 * parte / total


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n_arquivos, ocorrencias, tag_de, por_tipo_no_xml = ler_corpus(PASTA_XML)
    arestas = list(ocorrencias)
    total = len(arestas)

    print(f"Arquivos XML lidos: {n_arquivos}  [texto: 998]\n")

    print("1. Ocorrências de cada tipo de relação nos XMLs  [texto, cap. 2: 9.825 e 1.600]")
    for tipo, c in por_tipo_no_xml.most_common():
        print(f"   {tipo}: {c}")

    print(f"\n2. Arestas do grafo unificado: {total}  [texto, cap. 3: 9.253]")
    for tipo in ("associated_with", "negation_of"):
        n = sum(1 for (_, _, t) in arestas if t == tipo)
        print(f"   {tipo}: {n} ({pct(n, total):.1f}%)")
    print("   [texto: 8.365 (90,4%) e 888 (9,6%)]")

    print("\n3. Cardinalidade, direção e arestas inversas, por tipo de relação")
    print("   [texto: associated_with N-para-N, 7,4% com inversa; negation_of 1-para-N,")
    print("    3,6% com inversa e 96% partindo de uma expressão de negação]")
    for tipo in ("associated_with", "negation_of"):
        pares = [(o, d) for (o, d, t) in arestas if t == tipo]
        caudas = collections.defaultdict(set)
        cabecas = collections.defaultdict(set)
        for o, d in pares:
            caudas[o].add(d)
            cabecas[d].add(o)
        tph = sum(len(v) for v in caudas.values()) / len(caudas)
        hpt = sum(len(v) for v in cabecas.values()) / len(cabecas)
        classe = ("1" if hpt < 1.5 else "N") + "-para-" + ("1" if tph < 1.5 else "N")
        conjunto = set(pares)
        com_inversa = sum((d, o) in conjunto for o, d in pares)
        origem_negacao = sum("Negation" in (tag_de[o] or "") for o, d in pares)
        print(f"   {tipo}: {classe} (caudas por cabeça {tph:.2f}; cabeças por cauda {hpt:.2f};"
              f" critério de Bordes et al., limiar 1,5)")
        print(f"      com a aresta inversa: {pct(com_inversa, len(pares)):.1f}%"
              f" | origem marcada como Negation: {pct(origem_negacao, len(pares)):.1f}%")

    print("\n4. O que associated_with liga (pares de categoria mais frequentes)")
    print("   [texto: 464, 412 e 402; doença <-> sintoma em 40 arestas, 0,5%]")
    assoc = [(o, d) for (o, d, t) in arestas if t == "associated_with"]
    pares_cat = collections.Counter((categoria(tag_de[o]), categoria(tag_de[d])) for o, d in assoc)
    for (a, b), c in pares_cat.most_common(3):
        print(f"   {c:4d}  {a} -> {b}")
    doenca_sintoma = sum(c for (a, b), c in pares_cat.items()
                         if {a, b} == {"Disease or Syndrome", "Sign or Symptom"})
    print(f"   Disease or Syndrome <-> Sign or Symptom: {doenca_sintoma}"
          f" ({pct(doenca_sintoma, len(assoc)):.1f}% das associações)")

    print("\n5. Composição: caminhos a -> b -> c formados por associated_with")
    print("   [texto: 23.661 caminhos, 236 (1,0%) com a aresta direta a -> c]")
    sucessores = collections.defaultdict(set)
    for o, d in assoc:
        sucessores[o].add(d)
    caminhos = fechados = 0
    for a, bs in sucessores.items():
        for b in bs:
            for c in sucessores.get(b, ()):
                if c != a:
                    caminhos += 1
                    fechados += c in bs
    print(f"   caminhos: {caminhos} | com aresta direta: {fechados} ({pct(fechados, caminhos):.1f}%)")

    print("\n6. Quantos conceitos diferentes cada expressão de negação nega")
    print("   [texto: SEM 351, NEGA 211, NÃO 115]")
    negados = collections.defaultdict(set)
    for (o, d, t) in arestas:
        if t == "negation_of":
            negados[texto_de(o)].add(d)
    print(f"   expressões de negação distintas: {len(negados)}")
    for expressao, conceitos in sorted(negados.items(), key=lambda kv: -len(kv[1]))[:3]:
        print(f"   {expressao}: {len(conceitos)}")

    print("\n7. Exemplos de triplas usados no texto (vezes em que aparecem no corpus)")
    print("   [texto: LIMPO->CURATIVO 31; PUPILAS->FOTORREAGENTES 19; SEM->QUEIXAS 56;")
    print("    NEGA->TABAGISMO 19; SEM->EDEMA 25; OSTEOMIELITE->NÃO ESPECIFICADA 1]")
    exemplos = [("LIMPO", "associated_with", "CURATIVO"),
                ("PUPILAS", "associated_with", "FOTORREAGENTES"),
                ("SEM", "negation_of", "QUEIXAS"),
                ("NEGA", "negation_of", "TABAGISMO"),
                ("SEM", "negation_of", "EDEMA"),
                ("OSTEOMIELITE", "negation_of", "NÃO ESPECIFICADA")]
    for origem, tipo, destino in exemplos:
        encontrados = sorted((c for (o, d, t), c in ocorrencias.items()
                              if t == tipo and texto_de(o) == origem and texto_de(d) == destino),
                             reverse=True)
        print(f"   ({origem}, {tipo}, {destino}): {encontrados}"
              "  (mais de um valor = o mesmo texto anotado com categorias diferentes)")

    print("\n8. Quantas vezes cada aresta aparece no corpus  [texto: 89,8% aparecem uma única vez]")
    uma_vez = sum(1 for c in ocorrencias.values() if c == 1)
    print(f"   uma única vez: {uma_vez} de {total} ({pct(uma_vez, total):.1f}%)"
          f" | máximo: {max(ocorrencias.values())}")


if __name__ == "__main__":
    main()
