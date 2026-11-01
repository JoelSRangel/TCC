"""
Carrega no Neo4j o grafo do experimento 5g, para ver e consultar pelo Neo4j Browser.

O grafo das anotações (experimentos 1 a 4) fica no mesmo banco, em nós :Entidade ligados por :CONECTA_A, e o resto
do projeto só lê esses dois (o GDS projeta Entidade e CONECTA_A, e o cluster_benchmark.py lê os mesmos). O grafo do
5g entra separado e não mexe neles:
  - cada conceito é um nó com a categoria como rótulo (Doenca, Medicamento, SinalSintoma...), o texto mais comum em
    name (o campo que o Neo4j Browser escreve no desenho), a chave "texto normalizado|categoria" em id, as notas em
    que ele aparece e grafo = "5g";
  - cada ligação tem o tipo do 5g (TRATA, TEM_RESULTADO, LOCALIZADO_EM, NEGA, ASSOCIADO_A), completo = true quando
    ela está no grafo completo, filtrado = true quando está no filtrado e grafo = "5g".
Quase sempre as duas marcas andam juntas, mas o filtro do 5d desvira as ligações que vieram com a direção trocada
(RONCOS DIFUSOS tem_resultado AUSCULTA, no filtrado, vira AUSCULTA tem_resultado RONCOS DIFUSOS). Essas poucas
ficam no banco nas duas direções: a do completo e a do filtrado.
Cada carga apaga o 5g que já estava no banco e põe o novo, então dá para rodar de novo a cada parte que fica pronta.

Lê results/experimento5g/prontas/ (todas as partes prontas juntas) ou a pasta de uma parte. Fala com o Neo4j pela
API HTTP (porta 7474), com o usuário e a senha do .env (NEO4J_USER e NEO4J_PASSWORD, os mesmos do config.py), e por
isso não precisa do pacote neo4j. No fim, confere no banco o que foi carregado e mostra consultas para colar no Neo4j
Browser.

Uso:
    python -X utf8 experimentos/05g_carregar_no_neo4j.py                 # carrega prontas/
    python -X utf8 experimentos/05g_carregar_no_neo4j.py --pasta parte1  # só a parte 1
    python -X utf8 experimentos/05g_carregar_no_neo4j.py --testar        # só lê e confere os arquivos
    python -X utf8 experimentos/05g_carregar_no_neo4j.py --apagar        # tira o 5g do banco
"""
import argparse
import base64
import collections
import json
import os
import sys
import urllib.error
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento5g")
GRAFO = "5g"  # a marca dos nós e das ligações deste grafo no banco
LOTE = 1000  # nós ou ligações por pedido ao Neo4j

CONSULTAS = (
    ("um pedaço do grafo", "MATCH p=()-[r {completo: true}]->() RETURN p LIMIT 300"),
    ("um conceito e os vizinhos dele",
     'MATCH p=(c)-[r {completo: true}]-() WHERE c.grafo = "5g" AND toLower(c.name) CONTAINS "angina" RETURN p'),
    ("um pedaço do grafo filtrado", "MATCH p=()-[r {filtrado: true}]->() RETURN p LIMIT 300"),
    ("quantos conceitos de cada categoria",
     'MATCH (n) WHERE n.grafo = "5g" RETURN labels(n)[0] AS categoria, count(*) AS conceitos ORDER BY conceitos DESC'),
)


def rotulo(categoria):
    return "".join(p.capitalize() for p in categoria.split("_"))  # sinal_sintoma -> SinalSintoma


def ler_grafo(pasta):
    """Os nós (por rótulo) e as ligações (por tipo) dos grafos completo e filtrado da pasta, com as duas marcas."""
    def triplas(nome):
        with open(os.path.join(pasta, nome), encoding="utf-8") as f:
            return [tuple(linha.rstrip("\n").split("\t")) for linha in f if linha.strip()]

    if not os.path.exists(os.path.join(pasta, "grafo_completo.tsv")):
        sys.exit(f"Não há grafo em {os.path.relpath(pasta, RAIZ)}: rode antes a parte montar do 05g.")
    with open(os.path.join(pasta, "entidades.json"), encoding="utf-8") as f:
        entidades = json.load(f)
    completo, filtrado = triplas("grafo_completo.tsv"), triplas("grafo_filtrado.tsv")
    todas, completo, filtrado = list(dict.fromkeys(completo + filtrado)), set(completo), set(filtrado)
    nos, sem_dados = collections.defaultdict(list), 0
    for chave in dict.fromkeys(e for o, _, d in todas for e in (o, d)):
        texto, categoria = chave.rsplit("|", 1)
        e = entidades.get(chave)
        sem_dados += e is None
        no = {"id": chave, "name": e["texto"] if e else texto, "categoria": categoria, "grafo": GRAFO}
        if e:
            no.update(ocorrencias=e["ocorrencias"], notas=e["notas"])
        nos[rotulo(categoria)].append(no)
    ligacoes = collections.defaultdict(list)
    for o, tipo, d in todas:
        ligacoes[tipo.upper()].append({"origem": o, "destino": d, "completo": (o, tipo, d) in completo,
                                       "filtrado": (o, tipo, d) in filtrado})
    return nos, ligacoes, sem_dados


def credenciais():
    """Usuário e senha do Neo4j: do ambiente ou, se não estiverem lá, do .env da raiz (como faz o config.py)."""
    usuario, senha = os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD")
    caminho = os.path.join(RAIZ, ".env")
    if not (usuario and senha) and os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha.startswith("export "):
                    linha = linha[len("export "):]
                if linha and not linha.startswith("#") and "=" in linha:
                    chave, valor = (s.strip() for s in linha.split("=", 1))
                    valor = valor[1:-1] if len(valor) > 1 and valor[0] == valor[-1] and valor[0] in "'\"" else valor
                    if chave == "NEO4J_USER" and not usuario:
                        usuario = valor
                    elif chave == "NEO4J_PASSWORD" and not senha:
                        senha = valor
    if not senha:
        sys.exit("NEO4J_PASSWORD não definida: preencha a senha do seu Neo4j no .env da raiz (veja .env.example).")
    return usuario or "neo4j", senha


class Neo4j:
    """O pouco da API HTTP do Neo4j que a carga usa: uma consulta por pedido, cada uma na sua transação."""

    def __init__(self, http, banco):
        try:
            with urllib.request.urlopen(http, timeout=10) as resposta:
                descoberta = json.load(resposta)
        except (OSError, ValueError) as erro:
            sys.exit(f"O Neo4j não respondeu em {http} ({erro}). Ele está ligado?")
        self.versao = descoberta.get("neo4j_version", "?")
        self.edicao = descoberta.get("neo4j_edition", "?")
        modelo = descoberta.get("transaction", http.rstrip("/") + "/db/{databaseName}/tx")
        self.url = modelo.replace("{databaseName}", banco) + "/commit"
        # elementId() existe a partir do Neo4j 5 (as versões de 2025 em diante são numeradas pelo ano)
        self.funcao_id = "elementId" if self.versao.split(".")[0].isdigit() and int(self.versao.split(".")[0]) >= 5 \
            else "id"
        usuario, senha = credenciais()
        self.cabecalhos = {"Content-Type": "application/json", "Accept": "application/json;charset=UTF-8",
                           "Authorization": "Basic " + base64.b64encode(f"{usuario}:{senha}".encode()).decode()}

    def rodar(self, consulta, **parametros):
        corpo = json.dumps({"statements": [{"statement": consulta, "parameters": parametros}]}).encode("utf-8")
        pedido = urllib.request.Request(self.url, data=corpo, headers=self.cabecalhos, method="POST")
        try:
            with urllib.request.urlopen(pedido, timeout=300) as resposta:
                conteudo = json.load(resposta)
        except urllib.error.HTTPError as erro:
            sys.exit("O Neo4j recusou o usuário ou a senha do .env." if erro.code == 401
                     else f"O Neo4j respondeu com erro {erro.code} em {self.url}.")
        except OSError as erro:
            sys.exit(f"A conexão com o Neo4j caiu ({erro}).")
        if conteudo.get("errors"):
            sys.exit(f"O Neo4j recusou a consulta: {conteudo['errors'][0].get('message')}")
        resultado = conteudo["results"][0]
        return [dict(zip(resultado["columns"], linha["row"])) for linha in resultado["data"]]


def apagar(banco):
    """Tira do banco os nós do 5g e as ligações deles, aos poucos; os nós :Entidade nunca entram."""
    total = 0
    while True:
        apagados = banco.rodar("MATCH (n) WHERE n.grafo = $grafo AND NOT n:Entidade WITH n LIMIT $lote"
                               " DETACH DELETE n RETURN count(*) AS apagados", grafo=GRAFO, lote=5 * LOTE)[0]["apagados"]
        total += apagados
        if not apagados:
            return total


def carregar_no_banco(banco, nos, ligacoes):
    fid, ids = banco.funcao_id, {}
    for r, lista in sorted(nos.items()):
        for i in range(0, len(lista), LOTE):
            linhas = banco.rodar(f"UNWIND $nos AS dados CREATE (n:`{r}`) SET n = dados"
                                 f" RETURN dados.id AS id, {fid}(n) AS eid", nos=lista[i:i + LOTE])
            ids.update((linha["id"], linha["eid"]) for linha in linhas)
    for tipo, lista in sorted(ligacoes.items()):
        for i in range(0, len(lista), LOTE):
            lote = [{"a": ids[l["origem"]], "b": ids[l["destino"]], "completo": l["completo"], "filtrado": l["filtrado"]}
                    for l in lista[i:i + LOTE]]
            banco.rodar(f"UNWIND $ligacoes AS l MATCH (a) WHERE {fid}(a) = l.a MATCH (b) WHERE {fid}(b) = l.b"
                        f" CREATE (a)-[r:`{tipo}`]->(b) SET r.completo = l.completo, r.filtrado = l.filtrado,"
                        f" r.grafo = $grafo", ligacoes=lote, grafo=GRAFO)


def contar_no_banco(banco):
    """Conceitos, ligações do completo e ligações do filtrado que o 5g tem no banco."""
    nos = banco.rodar("MATCH (n) WHERE n.grafo = $grafo RETURN count(n) AS nos", grafo=GRAFO)[0]["nos"]
    ligacoes = banco.rodar("MATCH ()-[r]->() WHERE r.grafo = $grafo RETURN"
                           " sum(CASE WHEN r.completo THEN 1 ELSE 0 END) AS completo,"
                           " sum(CASE WHEN r.filtrado THEN 1 ELSE 0 END) AS filtrado", grafo=GRAFO)[0]
    return nos, ligacoes["completo"], ligacoes["filtrado"]


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--pasta", default="prontas",
                            help="pasta do grafo dentro de results/experimento5g (prontas, parte1, ...)")
    argumentos.add_argument("--http", default="http://localhost:7474", help="endereço HTTP do Neo4j")
    argumentos.add_argument("--banco", default="neo4j", help="nome do banco dentro do Neo4j")
    argumentos.add_argument("--testar", action="store_true", help="só lê e confere os arquivos, sem o Neo4j")
    argumentos.add_argument("--apagar", action="store_true", help="tira o grafo do 5g do banco")
    args = argumentos.parse_args()

    if args.apagar:
        banco = Neo4j(args.http, args.banco)
        print(f"Neo4j {banco.versao} ({banco.edicao}): {apagar(banco)} nós do 5g apagados, com as ligações deles")
        return

    pasta = os.path.join(PASTA, args.pasta)
    nos, ligacoes, sem_dados = ler_grafo(pasta)
    todas = [l for lista in ligacoes.values() for l in lista]
    nos_arquivos = (sum(map(len, nos.values())), sum(l["completo"] for l in todas), sum(l["filtrado"] for l in todas))
    so_filtrado = sum(not l["completo"] for l in todas)
    print(f"Grafo em {os.path.relpath(pasta, RAIZ)}: {nos_arquivos[0]} conceitos, {nos_arquivos[1]} ligações no"
          f" completo e {nos_arquivos[2]} no filtrado ({so_filtrado} delas desviradas pelo filtro)")
    print("   conceitos: " + ", ".join(f"{r} {len(l)}" for r, l in sorted(nos.items(), key=lambda x: -len(x[1]))))
    print("   ligações do completo: " + ", ".join(
        f"{t} {n}" for t, n in sorted(((t, sum(l["completo"] for l in lista)) for t, lista in ligacoes.items()),
                                      key=lambda x: -x[1])))
    if sem_dados:
        print(f"   {sem_dados} conceitos das ligações não estão no entidades.json (entram só com o texto da chave)")
    if args.testar:
        return

    banco = Neo4j(args.http, args.banco)
    print(f"Neo4j {banco.versao} ({banco.edicao}), banco {args.banco}")
    print(f"   {apagar(banco)} nós de uma carga anterior do 5g apagados")
    carregar_no_banco(banco, nos, ligacoes)
    no_banco = contar_no_banco(banco)
    if no_banco == nos_arquivos:
        print(f"   carregado e conferido: o banco tem os {nos_arquivos[0]} conceitos e as ligações dos arquivos")
    else:
        print(f"   ATENÇÃO: o banco ficou com {no_banco[0]} conceitos, {no_banco[1]} ligações no completo e"
              f" {no_banco[2]} no filtrado, e os arquivos têm {nos_arquivos[0]}, {nos_arquivos[1]} e {nos_arquivos[2]}")
    print(f"\nPara ver: abra {args.http}/browser/, entre com o seu usuário e senha e cole uma destas consultas:")
    for o_que, consulta in CONSULTAS:
        print(f"   {o_que}:\n      {consulta}")


if __name__ == "__main__":
    main()
