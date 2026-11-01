"""
Experimento 5d: conferência à mão de uma amostra das relações do grafo do texto.

As anotações do SemClinBr marcam só parte das relações de cada nota: a maioria
das relações que o modelo de linguagem criou no experimento 5d não está nelas
e pode estar certa ou errada. Para saber quantas estão certas, uma pessoa lê o
trecho da nota de cada relação de uma amostra sorteada e diz se a relação está
certa.

A amostra tem 100 relações do grafo filtrado e 100 das que só estão no grafo
completo, sorteadas entre as das notas fora do piloto e misturadas, para quem
confere não saber de qual grafo veio cada uma. A medida do grafo completo pesa
cada grupo pelo tamanho dele no grafo. Como a ordem é sorteada, as primeiras
linhas da planilha já são uma amostra menor dos dois grupos.

Na coluna "certa?" da planilha:
  s   certa: a nota diz isso, do jeito que está escrito
  i   certa, mas ao contrário (por exemplo, "120/80 → tem valor → PA")
  t   os dois termos têm a ver um com o outro, mas o tipo da ligação está errado
  n   errada: a nota não diz isso, ou um dos termos não está na nota
  (em branco: não sei)
No grafo filtrado, as relações de direção trocada entram invertidas; para elas,
a resposta que conta como certa nesse grafo é a i.

O script tem duas partes:
  sortear    monta a planilha (não escreve por cima de uma que já existe)
  conferir   lê a planilha preenchida e mede quantas relações estão certas

Saídas (não versionadas):
  results/experimento5d_relacoes_para_conferir.csv    a planilha, que abre no Excel
  results/experimento5d_relacoes_para_conferir.json   de onde veio cada linha
  results/experimento5d_conferencia.json              as medidas

Uso:
    python experimentos/05d_conferir_relacoes.py sortear
    # preencha a coluna "certa?" no Excel e salve, mantendo o formato CSV
    python experimentos/05d_conferir_relacoes.py conferir
"""
import argparse
import collections
import csv
import importlib.util
import io
import json
import math
import os
import random
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANILHA = os.path.join(RAIZ, "results", "experimento5d_relacoes_para_conferir.csv")
AMOSTRA = os.path.join(RAIZ, "results", "experimento5d_relacoes_para_conferir.json")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5d_conferencia.json")
SEMENTE = 0
POR_GRUPO = 100
GRUPOS = ("filtrado", "só no completo")
RESPOSTAS = ("s", "i", "t", "n")
COLUNAS = ["número", "nota", "ligação", "trecho da nota", "certa? (s/i/t/n)", "observação"]
MARGEM = 120  # caracteres antes e depois dos termos no trecho da nota


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


grafo = carregar("grafo", "05d_grafo_do_texto.py")
piloto, categorias = grafo.piloto, grafo.categorias


def ocorrencias(texto, termo):
    """(início, fim) de cada vez que o termo aparece no texto como palavra inteira, sem ligar para maiúsculas."""
    termo = " ".join(termo.split())
    if not termo:
        return []
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(termo)}(?!\w)", texto, re.IGNORECASE)]


def trecho(texto, origem, destino):
    """O trecho da nota em volta dos dois termos da relação, com os termos entre colchetes. Escolhe as duas
    ocorrências mais próximas uma da outra; se ficarem longe, mostra os dois pedaços."""
    texto = " ".join(texto.split())
    de_origem, de_destino = ocorrencias(texto, origem), ocorrencias(texto, destino)
    if de_origem and de_destino:
        marcas = list(min(((o, d) for o in de_origem for d in de_destino), key=lambda p: abs(p[0][0] - p[1][0])))
    else:
        marcas = (de_origem or de_destino)[:1]
    faltam = [termo for termo, achou in ((origem, de_origem), (destino, de_destino)) if not achou]
    aviso = "".join(f" («{termo}» não aparece igual no texto)" for termo in faltam)
    if not marcas:
        return aviso.strip()
    juntas = []  # um termo dentro do outro vira uma marca só
    for a, b in sorted(marcas):
        if juntas and a <= juntas[-1][1]:
            juntas[-1] = (juntas[-1][0], max(b, juntas[-1][1]))
        else:
            juntas.append((a, b))
    janelas = []
    for a, b in juntas:
        inicio, fim = max(0, a - MARGEM), min(len(texto), b + MARGEM)
        if janelas and inicio <= janelas[-1][1]:
            janelas[-1] = (janelas[-1][0], fim)
        else:
            janelas.append((inicio, fim))
    pedacos = []
    for inicio, fim in janelas:
        partes, posicao = [], inicio
        for a, b in juntas:
            if inicio <= a and b <= fim:
                partes += [texto[posicao:a], "[", texto[a:b], "]"]
                posicao = b
        partes.append(texto[posicao:fim])
        pedacos.append(("…" if inicio else "") + "".join(partes) + ("…" if fim < len(texto) else ""))
    return " ".join(pedacos) + aviso


def para_mostrar(caminho):
    """O caminho a partir da pasta do projeto, ou inteiro se estiver em outro disco."""
    try:
        return os.path.relpath(caminho, RAIZ)
    except ValueError:
        return caminho


def celula(texto):
    """O Excel lê como fórmula o que começa com estes sinais; um espaço antes faz ele ler como texto."""
    return " " + texto if texto[:1] in ("=", "+", "-", "@") else texto


def relacoes_do_grafo():
    """As relações das notas fora do piloto que entram no grafo completo, separadas entre as que também entram no
    grafo filtrado e as outras (as mesmas escolhas da parte montar do experimento 5d)."""
    do_piloto = {os.path.basename(a) for a in grafo.ner.divisao()[2]}
    marcadas = {}
    for k in range(1, grafo.PARTES + 1):
        with open(grafo.arquivo_conceitos(k), encoding="utf-8") as f:
            marcadas.update({n["arquivo"]: n for n in json.load(f)["notas"]})
    relacoes = {grupo: [] for grupo in GRUPOS}
    for nota, r in sorted(grafo.registros_de_relacoes().items()):
        if nota in do_piloto or nota not in marcadas or not grafo.em_dia(r, marcadas[nota]):
            continue
        conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in r["conceitos"]}
        grupos = {t: set() if c == "abreviatura" else {c} for t, c in conceitos.items()}
        for x in r["relacoes"]:
            if grafo.ponta(x["origem"], conceitos) == grafo.ponta(x["destino"], conceitos):
                continue  # uma relação de um conceito com ele mesmo não entra nos grafos
            situacao = categorias.situacao(x, grupos)
            grupo = "filtrado" if situacao in ("respeita", "direção trocada") else "só no completo"
            relacoes[grupo].append({"nota": nota, "origem": x["origem"], "relacao": x["relacao"],
                                    "destino": x["destino"], "situacao": situacao,
                                    "bate_com_as_anotacoes": x not in r["avaliacao"]["novas"]})
    return relacoes


def sortear(de_novo):
    if os.path.exists(PLANILHA) and not de_novo:
        sys.exit(f"{os.path.relpath(PLANILHA, RAIZ)} já existe e pode ter respostas. Para sortear outra amostra por"
                 " cima dela, use --de-novo.")
    relacoes = relacoes_do_grafo()
    rng = random.Random(SEMENTE)
    amostra = [{"grupo": grupo, **x} for grupo in GRUPOS
               for x in rng.sample(relacoes[grupo], min(POR_GRUPO, len(relacoes[grupo])))]
    rng.shuffle(amostra)  # mistura os dois grupos, para quem confere não saber de qual grafo veio cada relação
    with open(PLANILHA, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(COLUNAS)
        for numero, x in enumerate(amostra, 1):
            texto = grafo.ner.texto_cru(os.path.join(piloto.PASTA_XML, x["nota"]))
            ligacao = f"{x['origem']} → {x['relacao'].replace('_', ' ')} → {x['destino']}"
            escritor.writerow([numero, x["nota"], celula(ligacao), celula(trecho(texto, x["origem"], x["destino"])),
                               "", ""])
    grafo.salvar({"semente": SEMENTE, "por_grupo": POR_GRUPO,
                  "relacoes_no_grafo": {grupo: {"relacoes": len(relacoes[grupo]),
                                                "batem_com_as_anotacoes": sum(x["bate_com_as_anotacoes"]
                                                                              for x in relacoes[grupo])}
                                        for grupo in GRUPOS},
                  "linhas": [{"numero": numero, **x} for numero, x in enumerate(amostra, 1)]}, AMOSTRA)
    for grupo in GRUPOS:
        print(f"   {grupo}: {len(relacoes[grupo])} relações no grafo, {sum(x['grupo'] == grupo for x in amostra)}"
              f" sorteadas")
    print(f"Planilha para conferir: {os.path.relpath(PLANILHA, RAIZ)} ({len(amostra)} linhas)")


def ler_respostas(caminho):
    """As respostas da planilha por número da linha, e as linhas com uma resposta que não é s, i, t ou n. Aceita
    a planilha salva pelo Excel em UTF-8 ou em ANSI, com ponto e vírgula ou com vírgula."""
    with open(caminho, "rb") as f:
        bruto = f.read()
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("cp1252")
    primeira = texto.split("\n", 1)[0]
    leitor = csv.reader(io.StringIO(texto), delimiter=";" if primeira.count(";") >= primeira.count(",") else ",")
    next(leitor)
    respostas, invalidas = {}, []
    for linha in leitor:
        if not linha or not linha[0].strip().isdigit():
            continue
        numero = int(linha[0])
        r = linha[4].strip().lower() if len(linha) > 4 else ""
        r = {"sim": "s", "não": "n", "nao": "n", "?": ""}.get(r, r)
        if r in RESPOSTAS:
            respostas[numero] = r
        elif r:
            invalidas.append(numero)
    return respostas, invalidas


def proporcao(acertos, total):
    """Proporção, com o intervalo de 95% de Wilson."""
    if not total:
        return None
    z, p = 1.96, acertos / total
    centro = (p + z * z / (2 * total)) / (1 + z * z / total)
    meia = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return {"valor": p, "de": centro - meia, "ate": centro + meia, "respostas": total}


def ponderada(partes):
    """Proporção num conjunto de grupos sorteados separadamente. partes: (tamanho do grupo no conjunto, acertos e
    respostas na amostra) de cada grupo. Intervalo de 95% pela aproximação normal."""
    partes = [(tamanho, a, n) for tamanho, a, n in partes if tamanho]
    if not partes or any(n == 0 for _, _, n in partes):
        return None
    soma = sum(tamanho for tamanho, _, _ in partes)
    p = sum(tamanho / soma * a / n for tamanho, a, n in partes)
    erro = math.sqrt(sum((tamanho / soma) ** 2 * (a / n) * (1 - a / n) / n for tamanho, a, n in partes))
    return {"valor": p, "de": max(0.0, p - 1.96 * erro), "ate": min(1.0, p + 1.96 * erro),
            "respostas": sum(n for _, _, n in partes)}


def certa_no_filtrado(x, r):
    return (x["situacao"] == "respeita" and r == "s") or (x["situacao"] == "direção trocada" and r == "i")


def conferir(caminho):
    with open(AMOSTRA, encoding="utf-8") as f:
        amostra = json.load(f)
    respostas, invalidas = ler_respostas(caminho)
    linhas = [(x, respostas[x["numero"]]) for x in amostra["linhas"] if x["numero"] in respostas]
    if invalidas:
        print(f"Atenção: respostas que não são s, i, t ou n nas linhas {invalidas}; elas ficaram de fora")
    no_grafo = amostra["relacoes_no_grafo"]

    def do_grupo(grupo, cond=lambda x: True):
        return [(x, r) for x, r in linhas if x["grupo"] == grupo and cond(x)]

    def no_completo(teste, cond=lambda x: True, tamanho=lambda g: no_grafo[g]["relacoes"]):
        return ponderada([(tamanho(g), sum(teste(x, r) for x, r in do_grupo(g, cond)), len(do_grupo(g, cond)))
                          for g in GRUPOS])

    certa, tem_a_ver = (lambda x, r: r == "s"), (lambda x, r: r in ("s", "i", "t"))
    filtrado, fora = do_grupo("filtrado"), do_grupo("só no completo")
    batem = lambda x: x["bate_com_as_anotacoes"]
    medidas = {
        "certas": {
            "grafo completo": no_completo(certa),
            "grafo filtrado": proporcao(sum(certa_no_filtrado(x, r) for x, r in filtrado), len(filtrado)),
            "só no grafo completo": proporcao(sum(certa(x, r) for x, r in fora), len(fora)),
            "no completo, as que batem com as anotações": no_completo(
                certa, batem, lambda g: no_grafo[g]["batem_com_as_anotacoes"]),
            "no completo, as novas": no_completo(
                certa, lambda x: not batem(x), lambda g: no_grafo[g]["relacoes"] - no_grafo[g]["batem_com_as_anotacoes"])},
        "termos têm a ver (s, i ou t)": {
            "grafo completo": no_completo(tem_a_ver),
            "grafo filtrado": proporcao(sum(tem_a_ver(x, r) for x, r in filtrado), len(filtrado))}}
    respostas_por_tipo = collections.defaultdict(collections.Counter)
    for x, r in linhas:
        respostas_por_tipo[x["relacao"]][r] += 1
    grafo.salvar({"planilha": para_mostrar(caminho), "respondidas": len(linhas),
                  "linhas": len(amostra["linhas"]), "respostas_invalidas": invalidas,
                  "respostas": dict(collections.Counter(r for _, r in linhas)), "medidas": medidas,
                  "respostas_por_tipo_na_amostra": {t: dict(c) for t, c in sorted(respostas_por_tipo.items())}},
                 SAIDA_JSON)

    pct = piloto.pct
    print(f"\n== Conferência à mão: {len(linhas)} de {len(amostra['linhas'])} relações respondidas"
          f" ({len(filtrado)} do grafo filtrado, {len(fora)} só do completo)")
    print(f"   respostas: {dict(collections.Counter(r for _, r in linhas))}")
    for nome, grupo in medidas.items():
        print(f"\n   {nome:44s}{'fração':>8s}   intervalo de 95%")
        for rotulo, m in grupo.items():
            if m is None:
                print(f"   {rotulo:44s}{'':>8s}   (faltam respostas)")
            else:
                print(f"   {rotulo:44s}{pct(m['valor']):>8s}   {pct(m['de']).strip()} a {pct(m['ate']).strip()}"
                      f"  ({m['respostas']} respostas)")
    print("\n   na amostra, por tipo de ligação (sem pesar os grupos): certas / respondidas")
    for tipo, c in sorted(respostas_por_tipo.items(), key=lambda t: -sum(t[1].values())):
        print(f"   {tipo.replace('_', ' '):18s}{c['s']:4d} / {sum(c.values()):<4d}{pct(c['s'] / sum(c.values())):>8s}")
    print(f"\nMedidas em {os.path.relpath(SAIDA_JSON, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("sortear", "conferir"))
    argumentos.add_argument("--de-novo", action="store_true", help="sortear por cima de uma planilha que já existe")
    argumentos.add_argument("--planilha", default=PLANILHA, help="planilha preenchida (parte conferir)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "sortear":
        sortear(args.de_novo)
    else:
        conferir(args.planilha)


if __name__ == "__main__":
    main()
