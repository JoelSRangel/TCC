"""
Estrutura do grafo nas duas modelagens (seção 4.1 do texto do TCC).

Compara, a partir dos mesmos XMLs:
  - o grafo por menção (fase inicial): cada anotação é um nó separado;
  - o grafo unificado (atual): cada conceito, identificado por texto
    normalizado + categoria bruta, é um único nó (chave de src/extractor.py).

Os números do grafo unificado devem bater com results/estatisticas_grafo.json,
gerado por graph_stats.py a partir do Neo4j (17.897 nós, 9.253 arestas,
9.936 isolados, 10.838 componentes, maior componente com 5.633 nós,
grau máximo 281). O script confere isso no final, se o arquivo existir.

Uso (da raiz do repositório):
    python analises/03_estrutura_do_grafo.py [pasta_com_os_xml]
"""
import collections
import glob
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "raw")
ESTATISTICAS_NEO4J = os.path.join(RAIZ, "results", "estatisticas_grafo.json")

# Cópia dos valores de results/estatisticas_grafo.json da branch main, usada
# quando o arquivo não está na branch atual (ele só é versionado na main).
ESTATISTICAS_NEO4J_MAIN = {
    "total_nos": 17897, "total_arestas": 9253, "nos_isolados": 9936,
    "num_componentes_conexos": 10838, "tamanho_maior_componente": 5633, "grau_max": 281.0,
}


def chave(texto, tag):
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def ler_corpus(pasta):
    nos_mencao, arestas_mencao = set(), []
    nos_unif, arestas_unif = set(), set()
    ocorrencias_aresta = collections.Counter()
    for arquivo in sorted(glob.glob(os.path.join(pasta, "*.xml"))):
        raiz = ET.parse(arquivo).getroot()
        local = {}
        for anotacao in raiz.findall(".//annotation"):
            nos_mencao.add((arquivo, anotacao.get("id")))
            k = chave(anotacao.get("text"), anotacao.get("tag"))
            local[anotacao.get("id")] = k
            nos_unif.add(k)
        for rel in raiz.findall(".//rel"):
            a1, a2 = rel.get("annotation1"), rel.get("annotation2")
            if a1 in local and a2 in local:
                arestas_mencao.append(((arquivo, a1), (arquivo, a2)))
                aresta = (local[a1], local[a2], rel.get("reltype"))
                arestas_unif.add(aresta)
                ocorrencias_aresta[aresta] += 1
    return nos_mencao, arestas_mencao, nos_unif, list(arestas_unif), ocorrencias_aresta


def estrutura(nos, arestas):
    """Grau, nós isolados e componentes conexos (grafo tratado como não direcionado)."""
    grau = collections.Counter()
    pai = {n: n for n in nos}

    def raiz_de(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for aresta in arestas:
        a, b = aresta[0], aresta[1]
        grau[a] += 1
        grau[b] += 1
        ra, rb = raiz_de(a), raiz_de(b)
        if ra != rb:
            pai[ra] = rb
    componentes = collections.Counter(raiz_de(n) for n in nos)
    graus = [grau[n] for n in nos]
    com_aresta = [g for g in graus if g > 0]
    isolados = len(graus) - len(com_aresta)
    return {
        "nos": len(nos),
        "arestas": len(arestas),
        "isolados": isolados,
        "componentes": len(componentes),
        "componentes_sem_isolados": len(componentes) - isolados,
        "maior_componente": max(componentes.values()),
        "nos_com_aresta": len(com_aresta),
        "grau_medio_todos": sum(graus) / len(graus),
        "grau_medio_com_aresta": sum(com_aresta) / len(com_aresta),
        "grau_mediano": statistics.median(graus),
        "grau_max": max(graus),
        "grau": grau,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nos_m, arestas_m, nos_u, arestas_u, ocorrencias = ler_corpus(PASTA_XML)
    m = estrutura(nos_m, arestas_m)
    u = estrutura(nos_u, arestas_u)

    print("1. Tabela da seção 4.1  [texto: coluna 'Por menção' e coluna 'Unificado']")
    linhas = [
        ("Nós", f"{m['nos']}", f"{u['nos']}"),
        ("Arestas", f"{m['arestas']}", f"{u['arestas']}"),
        ("Nós isolados", f"{m['isolados']} ({100 * m['isolados'] / m['nos']:.1f}%)",
         f"{u['isolados']} ({100 * u['isolados'] / u['nos']:.1f}%)"),
        ("Componentes conexos (sem os isolados)", f"{m['componentes_sem_isolados']}",
         f"{u['componentes_sem_isolados']}"),
        ("Maior componente conexo", f"{m['maior_componente']} nós", f"{u['maior_componente']} nós"),
        ("Grau médio dos nós com alguma aresta", f"{m['grau_medio_com_aresta']:.2f}",
         f"{u['grau_medio_com_aresta']:.2f}"),
        ("Grau máximo", f"{m['grau_max']}", f"{u['grau_max']}"),
    ]
    print(f"   {'':40s} {'Por menção':>16s} {'Unificado':>16s}")
    for nome, a, b in linhas:
        print(f"   {nome:40s} {a:>16s} {b:>16s}")

    print("\n2. Outros números do texto da seção 4.1")
    uma_vez = sum(1 for c in ocorrencias.values() if c == 1)
    print(f"   arestas que aparecem uma única vez: {100 * uma_vez / len(ocorrencias):.1f}%  [texto: 89,8%]")
    print(f"   nós com alguma aresta: {u['nos_com_aresta']}  [texto: 7.961]")
    print(f"   maior componente / nós com aresta: {100 * u['maior_componente'] / u['nos_com_aresta']:.1f}%"
          "  [texto: 70,8%]")
    print(f"   grau médio considerando todos os nós: {u['grau_medio_todos']:.2f}"
          f" | mediana: {u['grau_mediano']}")

    print("\n3. Distribuição de grau dos nós com alguma aresta (grafo unificado)")
    print("   [texto: 64,6% com uma aresta, 17,6% com duas, três nós com mais de 100]")
    graus = [g for g in u["grau"].values() if g > 0]
    faixas = [(1, 1, "grau 1"), (2, 2, "grau 2"), (3, 9, "grau 3 a 9"),
              (10, 99, "grau 10 a 99"), (100, 10 ** 9, "grau 100 ou mais")]
    for minimo, maximo, nome in faixas:
        n = sum(1 for g in graus if minimo <= g <= maximo)
        print(f"   {nome:17s} {n:5d} nós ({100 * n / len(graus):.1f}%)")
    print("   nós de maior grau:")
    for conceito, g in u["grau"].most_common(5):
        texto, tag = conceito.split("|", 1)
        print(f"      {texto} [{tag}]: {g}")

    if os.path.exists(ESTATISTICAS_NEO4J):
        neo4j = json.load(open(ESTATISTICAS_NEO4J, encoding="utf-8"))
        origem = "results/estatisticas_grafo.json"
    else:
        neo4j = ESTATISTICAS_NEO4J_MAIN
        origem = "valores de results/estatisticas_grafo.json na branch main"
    comparacoes = [("total_nos", u["nos"]), ("total_arestas", u["arestas"]),
                   ("nos_isolados", u["isolados"]), ("num_componentes_conexos", u["componentes"]),
                   ("tamanho_maior_componente", u["maior_componente"]), ("grau_max", u["grau_max"])]
    print(f"\n4. Conferência com as estatísticas calculadas pelo Neo4j ({origem})")
    for campo, valor in comparacoes:
        igual = "ok" if float(neo4j[campo]) == float(valor) else "DIFERENTE"
        print(f"   {campo}: Neo4j {neo4j[campo]} | script {valor} -> {igual}")


if __name__ == "__main__":
    main()
