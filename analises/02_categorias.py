"""
Categorias semânticas do SemClinBr e arestas entre categorias.

Recalcula os números citados no cap. 3 (métricas de agrupamento: 922
combinações de categorias reduzidas a 85) e no cap. 4 (separação das
categorias e análise exploratória: só 6,5% das arestas ligam entidades da
mesma categoria).

A redução de categorias segue a mesma regra de normalize_tag() em
cluster_benchmark.py: remove repetições, descarta os marcadores
Abbreviation e Negation quando há outro tipo junto e fica com o primeiro
tipo restante.

Uso (da raiz do repositório):
    python analises/02_categorias.py [pasta_com_os_xml]
"""
import collections
import glob
import os
import sys
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "raw")
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tags_brutas = collections.Counter()
    tag_do_conceito = {}
    arestas = set()
    for arquivo in sorted(glob.glob(os.path.join(PASTA_XML, "*.xml"))):
        raiz = ET.parse(arquivo).getroot()
        local = {}
        for anotacao in raiz.findall(".//annotation"):
            tag = anotacao.get("tag")
            tags_brutas[tag] += 1
            k = chave(anotacao.get("text"), tag)
            local[anotacao.get("id")] = k
            tag_do_conceito[k] = tag
        for rel in raiz.findall(".//rel"):
            origem = local.get(rel.get("annotation1"))
            destino = local.get(rel.get("annotation2"))
            if origem and destino:
                arestas.add((origem, destino, rel.get("reltype")))

    categorias = {categoria(t) for t in tags_brutas}
    print("1. Categorias  [texto, cap. 3: 922 combinações reduzidas a 85 categorias]")
    print(f"   combinações brutas distintas no atributo tag: {len(tags_brutas)}")
    print(f"   categorias depois da redução: {len(categorias)}")
    exemplo = next(t for t in tags_brutas if t and "|" in t)
    print(f"   exemplo de combinação: {exemplo} -> {categoria(exemplo)}")

    print("\n2. Arestas que ligam entidades da mesma categoria")
    print("   [texto, cap. 4: 6,5% no total; 7,2% em associated_with; nenhuma em negation_of]")
    total = mesma_total = 0
    for tipo in ("associated_with", "negation_of"):
        do_tipo = [(o, d) for (o, d, t) in arestas if t == tipo]
        mesma = sum(categoria(tag_do_conceito[o]) == categoria(tag_do_conceito[d]) for o, d in do_tipo)
        total += len(do_tipo)
        mesma_total += mesma
        print(f"   {tipo}: {mesma} de {len(do_tipo)} ({100 * mesma / len(do_tipo):.1f}%)")
    print(f"   total: {mesma_total} de {total} ({100 * mesma_total / total:.1f}%)")


if __name__ == "__main__":
    main()
