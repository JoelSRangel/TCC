"""
Experimento 7: sugerir diagnósticos para uma nota nova.

Este é o uso dos modelos de vetores na vida real. A clínica treina o modelo com as notas que já tem e, quando chega
uma nota nova, o sistema mostra ao médico uma lista de diagnósticos a considerar:
  1. o BioBERTpt (experimento 5d) marca os conceitos da nota;
  2. ficam os sinais, sintomas, exames, medicamentos e procedimentos que a nota não nega;
  3. o modelo dá uma pontuação a cada doença que conhece, a partir desses conceitos;
  4. as 10 doenças com a maior pontuação são as sugestões, cada uma com os conceitos que mais pesaram.
O sistema não diagnostica: ele lembra possibilidades, e quem decide é o médico.

Avaliação. As notas são divididas nas mesmas 5 partes do experimento 5d. O modelo treina com as ligações de 4 partes,
e as notas da 5ª fazem o papel de notas novas; depois as partes trocam de lugar. Assim cada nota é nova uma vez. Os
conceitos das notas novas vêm do BioBERTpt que não viu aquela parte, como numa nota de verdade. As doenças escritas na
nota ficam escondidas do sistema, e o gabarito são as doenças que os anotadores marcaram na nota sem negar (com a forma
por extenso das siglas, quando eles a anotaram). As 20 notas do piloto e as 2 de exemplo do 5g treinam, mas não contam
na medida. Uma doença que também aparece na entrada (por exemplo, marcada como sintoma) não é escondida: ela sai do
gabarito daquela nota e não pode ser sugerida.

Medidas, por nota: se alguma doença do gabarito aparece entre as 1, 3, 5 e 10 primeiras sugestões; a fração das
doenças do gabarito achadas entre as 10; e se houve acerto fora das 10 doenças mais comuns, que são o que uma lista
fixa já acertaria. Duas grafias da mesma doença contam como uma sugestão só.

Quem compete:
  popularidade            sempre as doenças mais frequentes nas notas de treino
  coocorrência nas notas  as doenças que mais aparecem nas mesmas notas que os conceitos da nota nova
  consulta ao grafo       as doenças ligadas direto a esses conceitos no grafo
  TransE, ComplEx, RotatE treinados em cada grafo, com a configuração escolhida nos experimentos 1 e 6; a pontuação
                          de uma doença é a média, entre os conceitos da nota, da melhor pontuação das ligações que
                          fazem sentido (por exemplo, "conceito trata doença"), em desvios padrão por pergunta
  Node2Vec                os mesmos parâmetros do src/embeddings.py (64 dimensões, 10 caminhadas de 10 passos por nó,
                          sem direção), escrito aqui em PyTorch para treinar em cada parte sem subir o Neo4j; a
                          pontuação é o cosseno médio entre os conceitos da nota e a doença
Os resultados saem também só com sinais, sintomas e exames, sem medicamentos e procedimentos, que são pistas mais
fortes. A configuração dos modelos foi escolhida antes, numa divisão que incluía estas notas; é um vazamento pequeno, e
ela é a mesma em todas as partes.

Saídas (não versionadas; têm nomes de conceitos):
  results/experimento7/                 sugestões por nota, parte e modelo; modelo final; sugestões das notas novas
  results/experimento7_resultados.json  a tabela final

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/07_sugestao_de_diagnosticos.py avaliar --rapido   # testa o código numa parte, com 20 épocas
    python experimentos/07_sugestao_de_diagnosticos.py avaliar            # continua de onde parou, se for interrompido
    python experimentos/07_sugestao_de_diagnosticos.py comparar           # a tabela final, com o que já foi salvo
    python experimentos/07_sugestao_de_diagnosticos.py treinar            # treina o modelo final com todas as notas
    python experimentos/07_sugestao_de_diagnosticos.py sugerir nota.txt   # sugestões para uma nota nova (.txt ou .xml)
"""
import argparse
import collections
import functools
import glob
import importlib.util
import json
import logging
import os
import re
import statistics
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pykeen
import torch
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento7")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento7_resultados.json")
PASTA_GRAFOS = os.path.join(RAIZ, "results", "experimento5g", "prontas")
MODELO_DE_CONCEITOS = os.path.expanduser("~/tcc_modelos/experimento5d_parte1")
SUGESTOES = 10
POSICOES = (1, 3, 5, 10)
REAMOSTRAS = 2000

ENTRADAS = {"todos os conceitos": ("sinal_sintoma", "exame", "medicamento", "procedimento"),
            "só sinais, sintomas e exames": ("sinal_sintoma", "exame")}
# A negação vale até o fim da frase, no máximo a esta distância; estas palavras negam o que vem antes delas
ESCOPO_DA_NEGACAO = 60
NEGAM_PARA_TRAS = {"AUSENTE", "AUSENTES", "NEGATIVO", "NEGATIVA", "NEGATIVOS", "NEGATIVAS"}
FIM_DA_FRASE = re.compile(r"[.;:!?\n]")

# As perguntas feitas ao modelo para cada conceito da nota: (tipo de ligação, ponta em que a doença fica)
CONSULTAS_5G = (("associado_a", "destino"), ("associado_a", "origem"), ("trata", "destino"), ("tem_resultado", "destino"))
GRAFOS = {
    "anotacoes": {"nome": "anotações", "resultados": "experimento1_ampliado_resultados.json",
                  "consultas": (("associated_with", "destino"), ("associated_with", "origem"))},
    "5g_completo": {"nome": "5g completo", "resultados": "experimento6_completo_resultados.json",
                    "consultas": CONSULTAS_5G, "arquivo": "grafo_completo.tsv", "indice": 0},
    "5g_filtrado": {"nome": "5g filtrado", "resultados": "experimento6_filtrado_resultados.json",
                    "consultas": CONSULTAS_5G, "arquivo": "grafo_filtrado.tsv", "indice": 1},
}
MODELOS = {"TransE": ("TransE", 64), "ComplEx": ("ComplEx", 64), "RotatE": ("RotatE", 64)}
MODELO_FINAL = ("5g_completo", "RotatE")

# Node2Vec, com os valores de src/embeddings.py e os padrões do GDS (janela 10, 5 negativos, p = q = 1)
N2V_DIMENSAO, N2V_PASSOS, N2V_CAMINHADAS, N2V_JANELA, N2V_NEGATIVOS = 64, 10, 10, 10, 5
N2V_EPOCAS, N2V_LOTE, N2V_TAXA = 3, 4096, 0.01


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Treino e avaliação dos modelos: os do experimento 1; conceitos e ligações das notas: os dos experimentos 5
retreino = carregar("retreino", "01_retreino_kge.py")
g5 = carregar("g5", "05g_grafo_do_texto_27b.py")
bancada, piloto, grafo = g5.bancada, g5.piloto, g5.grafo
categorias, ner = grafo.categorias, grafo.ner


@functools.cache
def casa(a, b):
    return piloto.casa(a, b, True)


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- as notas
def trechos_negados(texto, spans):
    """Os trechos que a nota nega: os que vêm logo depois de uma palavra de negação, na mesma frase, ou logo antes
    de "ausente" e parecidas."""
    negados = set()
    for gatilho in (s for s in spans if s[2] == "negacao"):
        para_tras = piloto.normalizar(texto[gatilho[0]:gatilho[1]]) in NEGAM_PARA_TRAS
        for s in spans:
            depois = s[1] <= gatilho[0] if para_tras else gatilho[1] <= s[0]
            entre = texto[s[1]:gatilho[0]] if para_tras else texto[gatilho[1]:s[0]]
            if s[2] != "negacao" and depois and len(entre) <= ESCOPO_DA_NEGACAO and not FIM_DA_FRASE.search(entre):
                negados.add(s)
    return negados


def conceitos_da_nota(texto, spans):
    """Os textos normalizados dos conceitos que a nota não nega, por categoria."""
    negados, saida = trechos_negados(texto, spans), collections.defaultdict(set)
    for s in spans:
        conceito = piloto.normalizar(texto[s[0]:s[1]])
        if conceito and s not in negados:
            saida[s[2]].add(conceito)
    return saida


def diagnosticos_anotados(caminho):
    """As doenças que os anotadores marcaram na nota sem negar, cada uma com as suas formas (o texto e a sigla por
    extenso)."""
    raiz = ET.parse(caminho).getroot()
    negadas = {r.get("annotation2") for r in raiz.findall(".//rel") if r.get("reltype") == "negation_of"}
    saida = set()
    for a in raiz.findall(".//annotation"):
        if a.get("id") not in negadas and "doenca" in categorias.grupos(a.get("tag")):
            formas = {piloto.normalizar(a.get("text") or ""), piloto.normalizar(a.get("abbr") or "")} - {""}
            if formas:
                saida.add(frozenset(formas))
    return saida


def ler_notas():
    notas = {}
    for k in range(1, grafo.PARTES + 1):
        for item in g5.notas_da_parte(k):
            n = bancada.nota_marcada(item)
            notas[item["arquivo"]] = {"arquivo": item["arquivo"], "parte": k, "caminho": n["caminho"], "lista": n["lista"],
                                      "conceitos": conceitos_da_nota(n["texto"], n["spans"]),
                                      "diagnosticos": diagnosticos_anotados(n["caminho"])}
    return notas


def entradas(conceitos, variante):
    return sorted(set().union(*(conceitos.get(c, set()) for c in ENTRADAS[variante])))


def triplas_das_anotacoes(notas):
    por_nota = {}
    for arquivo, n in notas.items():
        raiz = ET.parse(n["caminho"]).getroot()
        local = {a.get("id"): retreino.chave(a.get("text"), a.get("tag")) for a in raiz.findall(".//annotation")}
        pares = ((local.get(r.get("annotation1")), r.get("reltype"), local.get(r.get("annotation2")))
                 for r in raiz.findall(".//rel"))
        por_nota[arquivo] = {t for t in pares if t[0] and t[2]}
    return por_nota


def triplas_do_texto(notas, g):
    with open(os.path.join(PASTA_GRAFOS, GRAFOS[g]["arquivo"]), encoding="utf-8") as f:
        validas = {tuple(linha.rstrip("\r\n").split("\t")) for linha in f if linha.strip()}
    registros, por_nota = g5.registros_de_relacoes(), {}
    for arquivo, n in notas.items():
        lista = {piloto.normalizar(c["texto"]): c["categoria"] for c in n["lista"]}
        por_nota[arquivo] = set(grafo.triplas_da_nota(registros[arquivo]["relacoes"], lista)[GRAFOS[g]["indice"]]) & validas
    return por_nota


def triplas_por_nota(notas, g):
    return triplas_das_anotacoes(notas) if g == "anotacoes" else triplas_do_texto(notas, g)


def texto_da(entidade):
    return piloto.normalizar(entidade.split("|", 1)[0])


def e_doenca(entidade):
    categoria = entidade.split("|", 1)[1]
    return "doenca" in categorias.grupos(categoria) or categoria == "doenca"


# ---------------------------------------------------------------- as pontuações
def indice_das_doencas(entidade_para_id):
    """Os textos das doenças do grafo e, em ordem, os ids das entidades de cada texto (e onde cada texto começa)."""
    por_texto = collections.defaultdict(list)
    for e, i in entidade_para_id.items():
        if e_doenca(e):
            por_texto[texto_da(e)].append(i)
    candidatos = sorted(por_texto)
    ids = np.array([i for t in candidatos for i in por_texto[t]])
    inicios = np.cumsum([0] + [len(por_texto[t]) for t in candidatos[:-1]])
    return candidatos, ids, inicios


def indice_das_entradas(entidade_para_id, textos):
    """Os ids das entidades que não são doenças, para cada texto de conceito pedido."""
    por_texto = collections.defaultdict(list)
    for e, i in entidade_para_id.items():
        if texto_da(e) in textos and not e_doenca(e):
            por_texto[texto_da(e)].append(i)
    return por_texto


def vetores_kge(modelo, relacao_para_id, consultas, ids_de, ids_doencas, inicios):
    """Para cada conceito, a pontuação de cada doença: a melhor entre as perguntas feitas ao modelo, em desvios padrão
    em relação às doenças daquela pergunta."""
    destino = [relacao_para_id[r] for r, lado in consultas if lado == "destino" and r in relacao_para_id]
    origem = [relacao_para_id[r] for r, lado in consultas if lado == "origem" and r in relacao_para_id]
    modelo.eval()
    saida = {}
    with torch.no_grad():
        for texto, ids in ids_de.items():
            linhas = []
            if destino:
                linhas.append(modelo.score_t(torch.tensor([[e, r] for e in ids for r in destino])))
            if origem:
                linhas.append(modelo.score_h(torch.tensor([[r, e] for e in ids for r in origem])))
            pontos = torch.cat(linhas)[:, torch.as_tensor(ids_doencas)].numpy()
            z = (pontos - pontos.mean(1, keepdims=True)) / (pontos.std(1, keepdims=True) + 1e-9)
            saida[texto] = np.maximum.reduceat(z.max(0), inicios)
    return saida


def treinar_kge(triplas, g, rotulo, semente, max_epocas, tentativas=3):
    """Treina com a configuração escolhida antes. Se a validação mostrar que o treino desandou (quase o acaso), como
    aconteceu com o ComplEx no experimento 6, troca a semente."""
    config = ler_json(os.path.join(RAIZ, "results", GRAFOS[g]["resultados"]))["resumo"][rotulo]["configuracao"]
    fabrica = TriplesFactory.from_labeled_triples(triples=np.array(sorted(triplas), dtype=str))
    treino, validacao = fabrica.split([0.9, 0.1], random_state=retreino.SEMENTE_DIVISAO)
    acaso = retreino.referencia_ao_acaso(treino.num_entities)["MRR"]
    nome, dimensao = MODELOS[rotulo]
    for tentativa in range(tentativas):
        modelo, info = retreino.treinar(nome, dimensao, config["perda"], config["negativos"], config["taxa"],
                                        semente + tentativa, treino, validacao, max_epocas)
        if (info["mrr_validacao"] or 0) >= 5 * acaso:
            break
    return modelo, treino, {**info, "configuracao": config, "tentativas": tentativa + 1}


def treinar_node2vec(triplas, semente, epocas=N2V_EPOCAS):
    """Vetores das entidades: caminhadas ao acaso pelo grafo, sem direção, e skip-gram com amostragem negativa."""
    entidades = sorted({x for o, _, d in triplas for x in (o, d)})
    indice = {e: i for i, e in enumerate(entidades)}
    vizinhos = [set() for _ in entidades]
    for o, _, d in triplas:
        vizinhos[indice[o]].add(indice[d])
        vizinhos[indice[d]].add(indice[o])
    grau = np.array([len(v) for v in vizinhos])
    inicio = np.concatenate([[0], np.cumsum(grau)[:-1]])
    adjacentes = np.array([x for v in vizinhos for x in sorted(v)])

    gerador = np.random.default_rng(semente)
    torch.manual_seed(semente)
    atual = np.tile(np.arange(len(entidades)), N2V_CAMINHADAS)
    caminhadas = [atual]
    for _ in range(N2V_PASSOS - 1):
        atual = adjacentes[inicio[atual] + gerador.integers(0, grau[atual])]
        caminhadas.append(atual)
    caminhadas = np.stack(caminhadas, axis=1)
    centros = np.concatenate([caminhadas[:, :-d].ravel() for d in range(1, N2V_JANELA)])
    contextos = np.concatenate([caminhadas[:, d:].ravel() for d in range(1, N2V_JANELA)])
    centros, contextos = np.concatenate([centros, contextos]), np.concatenate([contextos, centros])

    entrada = torch.nn.Embedding(len(entidades), N2V_DIMENSAO, sparse=True)
    saida = torch.nn.Embedding(len(entidades), N2V_DIMENSAO, sparse=True)
    torch.nn.init.normal_(entrada.weight, std=0.1)
    torch.nn.init.normal_(saida.weight, std=0.1)
    otimizador = torch.optim.SparseAdam([*entrada.parameters(), *saida.parameters()], lr=N2V_TAXA)
    pesos = torch.as_tensor(grau.astype(float) ** 0.75)
    for _ in range(epocas):
        ordem = gerador.permutation(len(centros))
        for i in range(0, len(ordem), N2V_LOTE):
            lote = ordem[i:i + N2V_LOTE]
            c, o = torch.as_tensor(centros[lote]), torch.as_tensor(contextos[lote])
            n = torch.multinomial(pesos, len(lote) * N2V_NEGATIVOS, replacement=True).view(len(lote), N2V_NEGATIVOS)
            vc = entrada(c)
            positivo = torch.nn.functional.logsigmoid((vc * saida(o)).sum(1))
            negativo = torch.nn.functional.logsigmoid(-(vc.unsqueeze(1) * saida(n)).sum(2)).sum(1)
            perda = -(positivo + negativo).mean()
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
    vetores = entrada.weight.detach().numpy()
    return indice, vetores / np.linalg.norm(vetores, axis=1, keepdims=True)


def vetores_node2vec(vetores, ids_de, ids_doencas, inicios):
    """Para cada conceito, o cosseno com cada doença (os vetores já têm comprimento 1)."""
    saida = {}
    for texto, ids in ids_de.items():
        saida[texto] = np.maximum.reduceat((vetores[ids] @ vetores[ids_doencas].T).max(0), inicios)
    return saida


def vetores_do_grafo(triplas, candidatos, textos):
    """Para cada conceito, em quantas ligações do grafo ele aparece com cada doença."""
    posicao = {c: i for i, c in enumerate(candidatos)}
    saida = {}
    for o, _, d in triplas:
        for a, b in ((texto_da(o), texto_da(d)), (texto_da(d), texto_da(o))):
            if a in textos and b in posicao:
                saida.setdefault(a, np.zeros(len(candidatos)))[posicao[b]] += 1
    return saida


def contagens(notas):
    """Em quantas notas cada doença aparece, em quantas cada conceito aparece, e cada par junto."""
    popularidade, ocorrencias, juntos = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for n in notas:
        doencas = n["conceitos"].get("doenca", set())
        conceitos = entradas(n["conceitos"], "todos os conceitos")
        popularidade.update(doencas)
        ocorrencias.update(conceitos)
        for c in conceitos:
            juntos[c].update(doencas)
    return popularidade, ocorrencias, juntos


def vetores_de_coocorrencia(candidatos, ocorrencias, juntos, textos):
    """Para cada conceito, a chance de cada doença aparecer numa nota que tem o conceito."""
    posicao = {c: i for i, c in enumerate(candidatos)}
    saida = {}
    for t in textos & set(juntos):
        v = np.zeros(len(candidatos))
        for d, n in juntos[t].items():
            if d in posicao:
                v[posicao[d]] = n / ocorrencias[t]
        saida[t] = v
    return saida


# ---------------------------------------------------------------- as sugestões
def escolher(pontos, candidatos, na_nota):
    """As doenças com a maior pontuação, sem o que já está na entrada e sem duas grafias da mesma doença."""
    sugeridas = []
    for i in np.argsort(-pontos, kind="stable"):
        c = candidatos[i]
        if any(casa(c, e) for e in na_nota) or any(casa(c, s) for s in sugeridas):
            continue
        sugeridas.append(c)
        if len(sugeridas) == SUGESTOES:
            break
    return sugeridas


def medir(sugeridas, ocultos, comuns):
    """Posição do primeiro acerto, fração das doenças do gabarito achadas e se houve acerto fora das mais comuns."""
    acertos = [any(casa(s, f) for g in ocultos for f in g) for s in sugeridas]
    return {"posicao": acertos.index(True) + 1 if any(acertos) else None,
            "cobertura": sum(any(casa(s, f) for s in sugeridas for f in g) for g in ocultos) / len(ocultos),
            "fora_dos_comuns": any(a and not any(casa(s, c) for c in comuns) for s, a in zip(sugeridas, acertos))}


def avaliar_metodo(teste, candidatos, popularidade, por_entrada, comuns):
    """por_entrada: {conceito: pontuação de cada doença}; sem ele, a sugestão é a popularidade."""
    desempate = np.array([popularidade[c] for c in candidatos], dtype=float)
    desempate = desempate / max(desempate.max(), 1)
    resultado = {}
    for variante in ENTRADAS:
        por_nota = {}
        for n in teste:
            na_nota = entradas(n["conceitos"], variante)
            ocultos = [g for g in n["diagnosticos"] if not any(casa(f, e) for f in g for e in na_nota)]
            if not ocultos:
                continue
            if por_entrada is None:
                sugeridas = escolher(desempate, candidatos, na_nota)
                conhecidos = len(na_nota)
            else:
                vetores = [por_entrada[e] for e in na_nota if e in por_entrada]
                conhecidos = len(vetores)
                sugeridas = escolher(np.mean(vetores, axis=0) + 1e-6 * desempate, candidatos, na_nota) if vetores else []
            por_nota[n["arquivo"]] = {**medir(sugeridas, ocultos, comuns), "sugestoes": sugeridas,
                                      "conceitos": len(na_nota), "conhecidos": conhecidos}
        resultado[variante] = por_nota
    return resultado


def gravar(arquivo, metodo, g, k, resultado, info=None):
    retreino.salvar_json({"metodo": metodo, "grafo": g, "parte": k, "info": info or {}, "resultado": resultado}, arquivo)
    pontos = [v["posicao"] is not None and v["posicao"] <= SUGESTOES for v in resultado["todos os conceitos"].values()]
    print(f"   {metodo:24s} {GRAFOS[g]['nome'] if g else '':12s} parte {k}: acerto entre as 10 {statistics.fmean(pontos):.1%}"
          f" em {len(pontos)} notas" + (f" ({info['segundos']} s)" if info and "segundos" in info else ""), flush=True)


def avaliar(args):
    pasta = os.path.join(PASTA, "rapido") if args.rapido else PASTA
    os.makedirs(pasta, exist_ok=True)
    notas = ler_notas()
    fora = set(ler_json(os.path.join(RAIZ, "results", "experimento5g_resultados.json"))["fora_das_medidas"])
    por_grafo = {g: triplas_por_nota(notas, g) for g in args.grafos}
    print(f"{len(notas)} notas; {sum(1 for n in notas.values() if n['diagnosticos'] and n['arquivo'] not in fora)} "
          f"contam na medida; ligações: " + ", ".join(f"{g} {len(set().union(*t.values()))}" for g, t in por_grafo.items()))

    for k in ([1] if args.rapido else range(1, grafo.PARTES + 1)):
        print(f"\n== Parte {k} como notas novas", flush=True)
        treino = [n for n in notas.values() if n["parte"] != k]
        teste = [n for n in notas.values() if n["parte"] == k and n["diagnosticos"] and n["arquivo"] not in fora]
        popularidade, ocorrencias, juntos = contagens(treino)
        textos = set().union(*(set(entradas(n["conceitos"], "todos os conceitos")) for n in teste))

        base = sorted(popularidade)
        ordem = np.array([popularidade[c] for c in base], dtype=float)
        comuns = escolher(ordem, base, [])
        for metodo, por_entrada in (("popularidade", None),
                                    ("coocorrência nas notas", vetores_de_coocorrencia(base, ocorrencias, juntos, textos))):
            arquivo = os.path.join(pasta, f"{metodo.split()[0]}_parte{k}.json")
            if not os.path.exists(arquivo):
                gravar(arquivo, metodo, None, k, avaliar_metodo(teste, base, popularidade, por_entrada, comuns))

        for g in args.grafos:
            triplas = set().union(*(t for a, t in por_grafo[g].items() if notas[a]["parte"] != k))

            arquivo = os.path.join(pasta, f"{g}_consulta_parte{k}.json")
            if not os.path.exists(arquivo):
                candidatos = indice_das_doencas({x: 0 for o, _, d in triplas for x in (o, d)})[0]
                por_entrada = vetores_do_grafo(triplas, candidatos, textos)
                gravar(arquivo, "consulta ao grafo", g, k, avaliar_metodo(teste, candidatos, popularidade, por_entrada, comuns))

            arquivo = os.path.join(pasta, f"{g}_Node2Vec_parte{k}.json")
            if not os.path.exists(arquivo) and "Node2Vec" in args.modelos:
                inicio = time.time()
                indice, vetores = treinar_node2vec(triplas, k, 1 if args.rapido else N2V_EPOCAS)
                candidatos, ids_doencas, inicios = indice_das_doencas(indice)
                por_entrada = vetores_node2vec(vetores, indice_das_entradas(indice, textos), ids_doencas, inicios)
                gravar(arquivo, "Node2Vec", g, k, avaliar_metodo(teste, candidatos, popularidade, por_entrada, comuns),
                       {"segundos": round(time.time() - inicio, 1)})

            for rotulo in (m for m in args.modelos if m in MODELOS):
                arquivo = os.path.join(pasta, f"{g}_{rotulo}_parte{k}.json")
                if os.path.exists(arquivo):
                    continue
                modelo, fabrica_treino, info = treinar_kge(triplas, g, rotulo, k, 20 if args.rapido else retreino.MAX_EPOCAS,
                                                           1 if args.rapido else 3)
                candidatos, ids_doencas, inicios = indice_das_doencas(fabrica_treino.entity_to_id)
                ids_de = indice_das_entradas(fabrica_treino.entity_to_id, textos)
                por_entrada = vetores_kge(modelo, fabrica_treino.relation_to_id, GRAFOS[g]["consultas"], ids_de,
                                          ids_doencas, inicios)
                gravar(arquivo, rotulo, g, k, avaliar_metodo(teste, candidatos, popularidade, por_entrada, comuns), info)
    if not args.rapido:
        comparar()


# ---------------------------------------------------------------- a tabela
def intervalo(valores, sorteios):
    """Intervalo de 95% da média, pelo bootstrap das notas."""
    baixo, alto = np.percentile(np.asarray(valores)[sorteios].mean(1), [2.5, 97.5])
    return float(baixo), float(alto)


def comparar():
    linhas = collections.defaultdict(dict)
    for arquivo in sorted(glob.glob(os.path.join(PASTA, "*_parte*.json"))):
        d = ler_json(arquivo)
        rotulo = d["metodo"] + (f" ({GRAFOS[d['grafo']]['nome']})" if d["grafo"] else "")
        for variante, por_nota in d["resultado"].items():
            linhas[variante].setdefault(rotulo, {}).update(por_nota)

    gerador = np.random.default_rng(0)
    saida = {}
    for variante, metodos in linhas.items():
        notas = sorted(set.intersection(*(set(m) for m in metodos.values())))
        sorteios = gerador.integers(0, len(notas), (REAMOSTRAS, len(notas)))
        print(f"\n== {variante}: {len(notas)} notas novas (acerto: alguma doença anotada entre as sugestões)")
        print(f"   {'':34s} {'1ª':>6s} {'3':>6s} {'5':>6s} {'10':>6s} {'intervalo de 95%':>16s} {'achadas':>8s} {'fora das comuns':>16s}")
        referencia = [metodos["coocorrência nas notas"][a] for a in notas] if "coocorrência nas notas" in metodos else None
        saida[variante] = {"notas": len(notas), "metodos": {}}
        ordem = sorted(metodos, key=lambda m: -statistics.fmean(
            metodos[m][a]["posicao"] is not None for a in notas))
        for m in ordem:
            r = [metodos[m][a] for a in notas]
            acerto = {k: [float(x["posicao"] is not None and x["posicao"] <= k) for x in r] for k in POSICOES}
            cobertura = [x["cobertura"] for x in r]
            fora = [float(x["fora_dos_comuns"]) for x in r]
            baixo, alto = intervalo(acerto[SUGESTOES], sorteios)
            print(f"   {m:34s} " + " ".join(f"{statistics.fmean(acerto[k]):6.1%}" for k in POSICOES)
                  + f" {f'{baixo:.1%} a {alto:.1%}':>16s} {statistics.fmean(cobertura):8.1%} {statistics.fmean(fora):16.1%}")
            linha = {"acerto": {str(k): statistics.fmean(acerto[k]) for k in POSICOES}, "intervalo_10": [baixo, alto],
                     "doencas_achadas": statistics.fmean(cobertura), "acerto_fora_das_comuns": statistics.fmean(fora),
                     "conceitos_conhecidos": statistics.fmean(x["conhecidos"] / max(x["conceitos"], 1) for x in r)}
            if referencia is not None:
                base = [float(x["posicao"] is not None) for x in referencia]
                dif = [a - b for a, b in zip(acerto[SUGESTOES], base)]
                linha["diferenca_para_coocorrencia"] = [statistics.fmean(dif), *intervalo(dif, sorteios)]
            saida[variante]["metodos"][m] = linha
    retreino.salvar_json(saida, SAIDA_JSON)
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


# ---------------------------------------------------------------- o uso numa nota nova
def caminho_final(g, rotulo, nome):
    return os.path.join(PASTA, "modelo_final", f"{g}_{rotulo}_{nome}")


def treinar_final(args):
    """Treina o modelo com todas as notas e guarda o que a parte sugerir precisa."""
    notas = ler_notas()
    triplas = set().union(*triplas_por_nota(notas, args.grafo).values())
    modelo, fabrica, info = treinar_kge(triplas, args.grafo, args.modelo, 0, retreino.MAX_EPOCAS)
    os.makedirs(os.path.dirname(caminho_final(args.grafo, args.modelo, "")), exist_ok=True)
    torch.save(modelo, caminho_final(args.grafo, args.modelo, "modelo.pt"))
    popularidade, _, _ = contagens(notas.values())
    retreino.salvar_json({"entidades": fabrica.entity_to_id, "relacoes": fabrica.relation_to_id,
                          "popularidade": popularidade, "info": info}, caminho_final(args.grafo, args.modelo, "mapas.json"))
    print(f"Modelo final: {len(triplas)} ligações, MRR de validação {info['mrr_validacao']:.4f}")


def marcar_conceitos(texto):
    """Os trechos que o BioBERTpt marca numa nota nova, como no experimento 5d."""
    import transformers

    transformers.logging.set_verbosity_error()
    tokenizador = transformers.AutoTokenizer.from_pretrained(ner.MODELO_BASE)
    modelo = transformers.AutoModelForTokenClassification.from_pretrained(MODELO_DE_CONCEITOS)
    modelo.eval()
    codigo = tokenizador(texto, add_special_tokens=False, return_offsets_mapping=True)
    spans = ner.prever(modelo, tokenizador, {"ids": codigo["input_ids"], "offsets": codigo["offset_mapping"]})
    return grafo.ajustar_as_palavras(texto, spans)


def sugerir(args):
    """As sugestões para uma nota nova. Grava num arquivo, porque a saída tem conceitos da nota."""
    mapas = ler_json(caminho_final(args.grafo, args.modelo, "mapas.json"))
    modelo = torch.load(caminho_final(args.grafo, args.modelo, "modelo.pt"), weights_only=False)
    xml = args.arquivo.endswith(".xml")
    if xml:
        texto = ner.texto_cru(args.arquivo)
    else:
        with open(args.arquivo, encoding="utf-8") as f:
            texto = f.read()
    inicio = time.time()
    conceitos = conceitos_da_nota(texto, marcar_conceitos(texto))
    na_nota = entradas(conceitos, "todos os conceitos")
    candidatos, ids_doencas, inicios = indice_das_doencas(mapas["entidades"])
    ids_de = indice_das_entradas(mapas["entidades"], set(na_nota))
    por_entrada = vetores_kge(modelo, mapas["relacoes"], GRAFOS[args.grafo]["consultas"], ids_de, ids_doencas, inicios)
    conhecidos = [e for e in na_nota if e in por_entrada]
    if not conhecidos:
        sys.exit("O modelo não conhece nenhum conceito desta nota.")
    pontos = np.mean([por_entrada[e] for e in conhecidos], axis=0)
    sugeridas = escolher(pontos, candidatos, na_nota)

    linhas = [f"Nota: {os.path.basename(args.arquivo)}",
              f"Conceitos usados: {len(conhecidos)} de {len(na_nota)} marcados ({time.time() - inicio:.1f} s)",
              "", f"Diagnósticos a considerar ({GRAFOS[args.grafo]['nome']}, {args.modelo}):"]
    for i, s in enumerate(sugeridas, 1):
        j = candidatos.index(s)
        pesos = sorted(conhecidos, key=lambda e: -por_entrada[e][j])[:3]
        linhas.append(f"  {i:2d}. {s}  (por causa de: {', '.join(pesos)})")
    if xml:
        ocultos = [g for g in diagnosticos_anotados(args.arquivo) if not any(casa(f, e) for f in g for e in na_nota)]
        achados = sum(any(casa(s, f) for s in sugeridas for f in g) for g in ocultos)
        linhas += ["", f"Doenças anotadas na nota: {len(ocultos)}; entre as sugestões: {achados}"]
    saida = args.saida or os.path.join(PASTA, f"sugestoes_{os.path.splitext(os.path.basename(args.arquivo))[0]}.txt")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print(f"Sugestões em {os.path.relpath(saida, RAIZ)} ({len(conhecidos)} conceitos usados, {time.time() - inicio:.1f} s)")
    if xml:
        print(f"Doenças anotadas: {len(ocultos)}; entre as sugestões: {achados}; primeira na posição "
              f"{next((i for i, s in enumerate(sugeridas, 1) if any(casa(s, f) for g in ocultos for f in g)), '—')}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    partes = argumentos.add_subparsers(dest="parte", required=True)
    a = partes.add_parser("avaliar", help="as notas de cada parte como notas novas")
    a.add_argument("--rapido", action="store_true", help="uma parte só, com 20 épocas: testa o código")
    a.add_argument("--grafos", nargs="+", choices=list(GRAFOS), default=list(GRAFOS))
    a.add_argument("--modelos", nargs="+", choices=[*MODELOS, "Node2Vec"], default=[*MODELOS, "Node2Vec"])
    partes.add_parser("comparar", help="a tabela final, com o que já foi salvo")
    for nome, ajuda in (("treinar", "treina o modelo final com todas as notas"), ("sugerir", "sugestões para uma nota nova")):
        p = partes.add_parser(nome, help=ajuda)
        p.add_argument("--grafo", choices=list(GRAFOS), default=MODELO_FINAL[0])
        p.add_argument("--modelo", choices=list(MODELOS), default=MODELO_FINAL[1])
        if nome == "sugerir":
            p.add_argument("arquivo", help="a nota, em .txt ou no .xml do SemClinBr")
            p.add_argument("--saida", help="arquivo com as sugestões (padrão: results/experimento7/)")
    args = argumentos.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)
    print(f"PyKEEN {pykeen.get_version()} | PyTorch {torch.__version__} | {args.threads} núcleos")
    {"avaliar": avaliar, "comparar": lambda _: comparar(), "treinar": treinar_final, "sugerir": sugerir}[args.parte](args)


if __name__ == "__main__":
    main()
