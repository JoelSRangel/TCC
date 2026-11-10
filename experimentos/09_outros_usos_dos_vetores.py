"""
Experimento 9: outros usos dos vetores, com os modelos do experimento 8.

O experimento 8 guardou os modelos de TransE, ComplEx, RotatE e Node2Vec treinados em 4 das 5 partes das notas, para os
dois grafos: o das anotações e o que a LLM montou. São 5 rodadas por grafo. Aqui esses modelos são usados, sem treinar
nada de novo, em três usos que o projeto ainda não tinha testado. Em cada um, as técnicas são comparadas com regras
simples, que não usam vetores.

  categoria   Prever o grupo do conceito (doença, medicamento, exame...) pelo vetor.
              Os conceitos são divididos em 5 partes: um classificador aprende com 4 e prevê o grupo dos conceitos da
              5ª. No grafo das anotações, o grupo vem da categoria que os especialistas deram (os 10 grupos do
              experimento 5b), e só entram os conceitos com um grupo só. No grafo da LLM, o grupo é o que o BioBERTpt
              marcou. São dois classificadores, porque cada técnica vai melhor com um: a regressão logística e o voto
              dos 10 conceitos mais parecidos (cosseno). Sem vetores, os mesmos classificadores recebem a lista dos
              vizinhos do conceito (com o tipo e a direção de cada ligação) ou a contagem dos grupos desses vizinhos;
              nessa contagem, o grupo dos conceitos que estão sendo previstos fica escondido. O piso é responder
              sempre o grupo mais comum.

  duplicatas  Achar o mesmo conceito guardado em duas entidades.
              No grafo, uma entidade é o texto mais a categoria. Por isso o mesmo texto aparece em mais de uma entidade
              quando recebeu categorias diferentes: são as "gêmeas", duplicatas de que temos o gabarito. Para cada
              entidade com gêmea, todas as outras são ordenadas pela semelhança dos vetores (cosseno), e a medida é se
              uma gêmea aparece em 1º lugar ou entre as 10 primeiras. A regra sem vetores ordena pelo número de vizinhos
              em comum, com os empates sorteados. O texto, é claro, acharia todas essas gêmeas; o teste serve para saber
              se o vetor reconhece o mesmo conceito, que é o que seria preciso para achar grafias diferentes.

  erros       Apontar as ligações suspeitas de um grafo já montado (só o grafo da LLM, que é o que tem erros).
              O modelo dá nota a cada ligação: a posição do destino entre os conceitos, para aquela origem e aquele
              tipo, e a da origem, para aquele destino, sem contar as outras ligações já conhecidas; vale a média das
              duas. As ligações de nota mais baixa seriam as primeiras a revisar. Um modelo decora as ligações com que
              treinou e dá nota máxima a quase todas, então a medida usa as 10% que o treino de cada rodada separou
              para saber quando parar: o modelo não aprendeu com elas. O Node2Vec fica de fora, porque treinou com
              todas. Gabarito: a ligação bate com as anotações da nota de onde saiu, como no experimento 8 (as notas
              do piloto e as de exemplo não contam). Regras sem vetores: popularidade, vizinhos em comum e apoio, que
              é o número de notas em que a LLM propôs a mesma ligação.
              Medidas: AUC (a chance de uma ligação que bate ter nota maior que uma que não bate; 0,5 é o acaso), com
              intervalo de 95% sorteando notas inteiras, e quantas batem entre as 25% de menor e de maior nota. Sai
              também o AUC nas ligações do treino, para mostrar que ali a nota não serve.

Cada número é a média das 5 rodadas. As rodadas compartilham a maior parte das notas, então o desvio entre elas é menor
do que seria com dados novos.

Saída (não versionada; só números):
  results/experimento9_resultados.json

Uso, no ambiente conda do projeto (no WSL), depois do experimento 8 (avaliar e avaliar --base llm):
    conda activate tcc_graphs
    python experimentos/09_outros_usos_dos_vetores.py tudo
    python experimentos/09_outros_usos_dos_vetores.py categoria             # ou duplicatas, ou erros
    python experimentos/09_outros_usos_dos_vetores.py tudo --rodadas 1      # testa o código numa rodada só
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import warnings

import numpy as np
import torch
from pykeen.triples import TriplesFactory
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento9_resultados.json")
DOBRAS = 5
CLASSIFICADORES = ("regressão", "10 mais parecidos")
PISO = "grupo mais comum"
SEM_GRUPO = ("sem_categoria", "abreviatura")
FAIXAS_DE_GRAU = (("1 vizinho", 1, 1), ("2 ou 3 vizinhos", 2, 3), ("4 ou mais vizinhos", 4, math.inf))
POSICOES = (1, 10)
KGE = ("TransE", "ComplEx", "RotatE")
LOTE = 256


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Grafos, modelos salvos e gabarito: os do experimento 8; preparação dos vetores: a do experimento 3
atualizacao = carregar("atualizacao", "08_atualizacao_do_grafo.py")
agrupamento = carregar("agrupamento", "03_agrupamento.py")
sugestao, retreino, texto_da = atualizacao.sugestao, atualizacao.retreino, atualizacao.texto_da
BASES, TECNICAS, REAMOSTRAS = atualizacao.BASES, atualizacao.TECNICAS, atualizacao.REAMOSTRAS


# ---------------------------------------------------------------- os grafos e os modelos de cada rodada
def grafo_da_rodada(k, notas, por_nota):
    """As ligações do grafo da rodada k (as das notas das outras 4 partes), as entidades em ordem e os vizinhos."""
    triplas = set().union(*(t for a, t in por_nota.items() if notas[a]["parte"] != k))
    entidades = sorted({x for o, _, d in triplas for x in (o, d)})
    indice = {e: i for i, e in enumerate(entidades)}
    vizinhos = [set() for _ in entidades]
    for o, _, d in triplas:
        vizinhos[indice[o]].add(indice[d])
        vizinhos[indice[d]].add(indice[o])
    return triplas, entidades, indice, vizinhos


def modelo_salvo(base, tecnica, k, entidades):
    """Os vetores de uma técnica na rodada k, preparados como no experimento 3 e na ordem de `entidades`; e, para
    TransE, ComplEx e RotatE, o modelo inteiro com os seus mapas."""
    pasta = os.path.join(BASES[base]["pasta"], "modelos")
    if tecnica == "Node2Vec":
        dados = np.load(os.path.join(pasta, f"Node2Vec_parte{k}.npz"))
        ids, vetores, modelo, mapas = [str(e) for e in dados["ids"]], dados["vetores"], None, None
    else:
        mapas = sugestao.ler_json(os.path.join(pasta, f"{tecnica}_parte{k}_mapas.json"))
        modelo = torch.load(os.path.join(pasta, f"{tecnica}_parte{k}.pt"), weights_only=False)
        modelo.eval()
        ids = sorted(mapas["entidades"], key=mapas["entidades"].get)
        vetores = modelo.entity_representations[0](indices=None).detach().cpu().numpy()
    posicao = {e: i for i, e in enumerate(ids)}
    assert set(posicao) == set(entidades), "o modelo salvo não é o deste grafo: rode de novo o experimento 8"
    return agrupamento.preparar(vetores)[[posicao[e] for e in entidades]].astype(np.float32), modelo, mapas


def resumir(valores):
    return {"media": statistics.fmean(valores), "desvio": statistics.pstdev(valores) if len(valores) > 1 else 0.0}


def resumir_rodadas(rodadas):
    """A média e o desvio, entre as rodadas, de cada medida (dicionários com as mesmas chaves, até 2 níveis)."""
    saida = {}
    for chave, valor in rodadas[0].items():
        if isinstance(valor, dict):
            saida[chave] = resumir_rodadas([r[chave] for r in rodadas])
        else:
            saida[chave] = resumir([r[chave] for r in rodadas])
    return saida


# ---------------------------------------------------------------- categoria
def grupo_da(entidade, base):
    """O grupo do conceito, ou None quando não há um só (várias categorias, ou nenhuma)."""
    categoria = entidade.split("|", 1)[1]
    if base == "anotacoes":
        grupos = sugestao.categorias.grupos(categoria)
        return next(iter(grupos)) if len(grupos) == 1 else None
    return None if categoria in SEM_GRUPO else categoria


def ligacoes_de_cada_entidade(triplas, indice, relacoes):
    """Uma linha por ponta de ligação: a entidade, o tipo com a direção (2 por tipo de ligação) e o vizinho."""
    linhas, tipos, outros = [], [], []
    for o, r, d in triplas:
        linhas += [indice[o], indice[d]]
        tipos += [2 * relacoes[r], 2 * relacoes[r] + 1]
        outros += [indice[d], indice[o]]
    return np.array(linhas), np.array(tipos), np.array(outros)


def medidas_de_categoria(previsto, y, grau):
    saida = {"acertos": float(np.mean(previsto == y)), "f1": float(f1_score(y, previsto, average="macro"))}
    for nome, minimo, maximo in FAIXAS_DE_GRAU:
        faixa = (grau >= minimo) & (grau <= maximo)
        saida[nome] = float(np.mean(previsto[faixa] == y[faixa]))
    return saida


def classificador(nome, padronizar):
    if nome == "regressão":
        regressao = LogisticRegression(max_iter=3000)
        return make_pipeline(StandardScaler(), regressao) if padronizar else regressao
    return KNeighborsClassifier(10, metric="cosine", algorithm="brute", weights="distance")


def categoria_na_rodada(base, k, triplas, entidades, indice, vizinhos):
    grupos = [grupo_da(e, base) for e in entidades]
    nomes = sorted({g for g in grupos if g})
    y_todas = np.array([nomes.index(g) if g else -1 for g in grupos])
    com_grupo = np.flatnonzero(y_todas >= 0)
    y, grau = y_todas[com_grupo], np.array([len(v) for v in vizinhos])[com_grupo]
    dobras = list(StratifiedKFold(DOBRAS, shuffle=True, random_state=k).split(com_grupo, y))

    relacoes = {r: i for i, r in enumerate(sorted({r for _, r, _ in triplas}))}
    linhas, tipos, outros = ligacoes_de_cada_entidade(triplas, indice, relacoes)
    n, uns = len(entidades), np.ones(len(linhas))
    lista = sparse.csr_matrix((uns, (linhas, tipos * n + outros)), shape=(n, 2 * len(relacoes) * n))
    lista = lista[:, np.flatnonzero(np.asarray(lista.sum(axis=0)).ravel() >= 2)]  # vizinho de um conceito só não ensina nada

    def grupos_dos_vizinhos(escondidos):
        visiveis = y_todas.copy()
        visiveis[escondidos] = -1
        coluna = tipos * (len(nomes) + 1) + np.where(visiveis[outros] >= 0, visiveis[outros], len(nomes))
        return sparse.csr_matrix((uns, (linhas, coluna)), shape=(n, 2 * len(relacoes) * (len(nomes) + 1)))

    def prever(entrada, padronizar):
        """As medidas de cada classificador com esta entrada (uma função que devolve a tabela, dados os escondidos)."""
        saida = {}
        for nome in CLASSIFICADORES:
            previsto = np.empty(len(y), dtype=int)
            for treino, teste in dobras:
                x = entrada(com_grupo[teste])
                modelo = classificador(nome, padronizar)
                modelo.fit(x[com_grupo[treino]], y[treino])
                previsto[teste] = modelo.predict(x[com_grupo[teste]])
            saida[nome] = medidas_de_categoria(previsto, y, grau)
        return saida

    mais_comum = np.empty(len(y), dtype=int)
    for treino, teste in dobras:
        mais_comum[teste] = np.bincount(y[treino]).argmax()
    saida = {PISO: {PISO: medidas_de_categoria(mais_comum, y, grau)},
             "lista dos vizinhos": prever(lambda _: lista, False),
             "grupos dos vizinhos": prever(grupos_dos_vizinhos, False)}
    for tecnica in TECNICAS:
        x = modelo_salvo(base, tecnica, k, entidades)[0]
        saida[tecnica] = prever(lambda _: x, True)
    return saida, {"conceitos": len(y), "grupos": len(nomes)}


# ---------------------------------------------------------------- duplicatas
def duplicatas_na_rodada(base, k, entidades, vizinhos):
    por_texto = collections.defaultdict(list)
    for i, e in enumerate(entidades):
        if texto_da(e):
            por_texto[texto_da(e)].append(i)
    gemeas = {i: [j for j in grupo if j != i] for grupo in por_texto.values() if len(grupo) > 1 for i in grupo}
    consultas = np.array(sorted(gemeas))
    n = len(entidades)
    e_gemea = np.zeros((len(consultas), n), dtype=bool)
    for linha, i in enumerate(consultas):
        e_gemea[linha, gemeas[i]] = True
    com_2 = np.array([len(vizinhos[i]) >= 2 for i in consultas])

    def medir(pontos):
        """A posição da gêmea mais bem colocada de cada consulta, com a própria entidade fora da lista."""
        pontos[np.arange(len(consultas)), consultas] = -np.inf
        melhor = np.where(e_gemea, pontos, -np.inf).max(axis=1)
        posicao = (pontos > melhor[:, None]).sum(axis=1) + 1
        saida = {f"entre as {p}": float(np.mean(posicao <= p)) for p in POSICOES}
        saida.update({f"entre as {p}, com 2 vizinhos ou mais": float(np.mean(posicao[com_2] <= p)) for p in POSICOES})
        return saida

    # Ao acaso: a chance de alguma das g gêmeas cair entre as p primeiras das n - 1 outras entidades
    def ao_acaso(p, selecao):
        return float(np.mean([1 - math.prod((n - 1 - len(gemeas[i]) - j) / (n - 1 - j) for j in range(p))
                              for i in consultas[selecao]]))
    todas = np.ones(len(consultas), dtype=bool)
    saida = {"ao acaso": {**{f"entre as {p}": ao_acaso(p, todas) for p in POSICOES},
                          **{f"entre as {p}, com 2 vizinhos ou mais": ao_acaso(p, com_2) for p in POSICOES}}}

    linhas = np.array([i for i, v in enumerate(vizinhos) for _ in v])
    colunas = np.array([j for v in vizinhos for j in v])
    adjacencia = sparse.csr_matrix((np.ones(len(linhas), dtype=np.float32), (linhas, colunas)), shape=(n, n))
    em_comum = (adjacencia[consultas] @ adjacencia.T).toarray()
    saida["vizinhos em comum"] = medir(em_comum + 0.5 * np.random.default_rng(k).random(em_comum.shape, dtype=np.float32))
    for tecnica in TECNICAS:
        x = modelo_salvo(base, tecnica, k, entidades)[0]
        saida[tecnica] = medir(x[consultas] @ x.T)
    return saida, {"consultas": len(consultas), "consultas com 2 vizinhos ou mais": int(com_2.sum()), "entidades": n}


# ---------------------------------------------------------------- erros
def notas_filtradas(modelo, mapas, triplas, grafo):
    """A nota de cada ligação: a posição do destino (para a origem e o tipo) e a da origem (para o tipo e o destino)
    entre os conceitos que não são outras ligações conhecidas do grafo; vale a média das duas, e 1 é o primeiro lugar."""
    e, r = mapas["entidades"], mapas["relacoes"]
    destinos, origens = collections.defaultdict(set), collections.defaultdict(set)
    for o, t, d in grafo:
        destinos[(e[o], r[t])].add(e[d])
        origens[(r[t], e[d])].add(e[o])
    ids = np.array([(e[o], r[t], e[d]) for o, t, d in triplas])
    saida, n = np.zeros(len(triplas)), len(e)
    with torch.no_grad():
        for colunas, pontuar, alvo, conhecidos in (((0, 1), modelo.score_t, 2, destinos), ((1, 2), modelo.score_h, 0, origens)):
            pares, de_par = np.unique(ids[:, colunas], axis=0, return_inverse=True)
            de_par = de_par.ravel()
            ordem = np.argsort(de_par, kind="stable")
            limites = np.searchsorted(de_par[ordem], np.arange(len(pares) + 1))
            for inicio in range(0, len(pares), LOTE):
                pontos = pontuar(torch.as_tensor(pares[inicio:inicio + LOTE])).numpy()
                for j in range(pontos.shape[0]):
                    daqui = ordem[limites[inicio + j]:limites[inicio + j + 1]]
                    s, alvos = pontos[j], ids[daqui, alvo]
                    sabidos = np.fromiter(conhecidos[tuple(int(v) for v in pares[inicio + j])], dtype=int)
                    na_frente = ((s[None, :] > s[alvos][:, None]).sum(axis=1)
                                 - (s[sabidos][None, :] > s[alvos][:, None]).sum(axis=1))
                    saida[daqui] += (1 - na_frente / (n - len(sabidos) + np.isin(alvos, sabidos))) / 2
    return saida


def erros_na_rodada(k, notas, propostas, certas, avaliadas):
    """As notas de cada técnica e de cada regra para as ligações do grafo da LLM na rodada k, em dois conjuntos: as que
    o treino separou e as que ele usou. Cada item é uma ligação numa nota."""
    triplas, entidades, indice, vizinhos = grafo_da_rodada(k, notas, propostas)
    fabrica = TriplesFactory.from_labeled_triples(triples=np.array(sorted(triplas), dtype=str))
    treino, separadas = fabrica.split([0.9, 0.1], random_state=retreino.SEMENTE_DIVISAO)  # a mesma divisão do treino
    apoio = collections.Counter(t for a, ts in propostas.items() if notas[a]["parte"] != k for t in ts)
    modelos = {}
    for tecnica in KGE:
        modelos[tecnica] = modelo_salvo("llm", tecnica, k, entidades)[1:]
        assert modelos[tecnica][1]["entidades"] == treino.entity_to_id and modelos[tecnica][1]["relacoes"] == treino.relation_to_id

    saida = {"ligacoes_no_grafo": len(triplas)}
    for nome, parte in (("separadas", separadas), ("do treino", treino)):
        lista = sorted({tuple(t) for t in parte.triples})
        posicao = {t: i for i, t in enumerate(lista)}
        por_metodo = {
            "apoio": np.array([apoio[t] for t in lista], dtype=float),
            "popularidade": np.array([len(vizinhos[indice[o]]) * len(vizinhos[indice[d]]) for o, _, d in lista], dtype=float),
            "vizinhos em comum": np.array([len(vizinhos[indice[o]] & vizinhos[indice[d]]) for o, _, d in lista], dtype=float)}
        for tecnica, (modelo, mapas) in modelos.items():
            por_metodo[tecnica] = notas_filtradas(modelo, mapas, lista, triplas)
        de_nota, certa, da_tripla = [], [], []
        for i, a in enumerate(avaliadas):
            if notas[a]["parte"] == k:
                continue
            for t in sorted(propostas[a]):
                if t in posicao:
                    de_nota.append(i)
                    certa.append(certas[a][t])
                    da_tripla.append(posicao[t])
        da_tripla = np.array(da_tripla)
        saida[nome] = {"de_nota": np.array(de_nota), "certa": np.array(certa), "ligacoes": len(lista),
                       "notas": {m: v[da_tripla] for m, v in por_metodo.items()},
                       "nota_maxima": {t: float(np.mean(por_metodo[t] >= 0.9999)) for t in KGE}}
    return saida


def parte_erros(rodadas):
    notas = sugestao.ler_notas()
    fora = set(sugestao.ler_json(os.path.join(RAIZ, "results", "experimento5g_resultados.json"))["fora_das_medidas"])
    propostas = sugestao.triplas_do_texto(notas, atualizacao.GRAFO_DA_LLM)
    avaliadas = sorted(a for a in notas if a not in fora)
    certas = {}
    for a in avaliadas:
        gabarito = atualizacao.pares_anotados(notas[a]["caminho"])
        certas[a] = {t: bool(atualizacao.combina(texto_da(t[0]), texto_da(t[2]), gabarito)) for t in propostas[a]}

    por_rodada = []
    for k in rodadas:
        inicio = time.time()
        por_rodada.append(erros_na_rodada(k, notas, propostas, certas, avaliadas))
        r = por_rodada[-1]
        print(f"   rodada {k}: {r['ligacoes_no_grafo']} ligações no grafo; separadas: {r['separadas']['ligacoes']} "
              f"({len(r['separadas']['certa'])} itens); do treino: {r['do treino']['ligacoes']} ({time.time() - inicio:.0f} s)", flush=True)

    separadas = [r["separadas"] for r in por_rodada]
    metodos = list(separadas[0]["notas"])
    gerador = np.random.default_rng(0)

    def auc_medio(conjuntos, selecoes):
        """A média, entre as rodadas, do AUC de cada método nos itens escolhidos de cada rodada."""
        return {m: statistics.fmean(atualizacao.auc(r["certa"][s], r["notas"][m][s]) for r, s in zip(conjuntos, selecoes))
                for m in metodos}

    todos = [slice(None)] * len(por_rodada)
    valores = auc_medio(separadas, todos)
    itens_da_nota = [[np.flatnonzero(r["de_nota"] == i) for i in range(len(avaliadas))] for r in separadas]
    amostras = collections.defaultdict(list)
    for sorteio in gerador.integers(0, len(avaliadas), (REAMOSTRAS, len(avaliadas))):
        for m, v in auc_medio(separadas, [np.concatenate([itens[i] for i in sorteio]) for itens in itens_da_nota]).items():
            amostras[m].append(v)
    amostras = {m: np.array(v) for m, v in amostras.items()}
    melhor_regra = max((m for m in metodos if m not in KGE), key=valores.get)

    saida = {"itens_por_rodada": statistics.fmean(len(r["certa"]) for r in separadas),
             "batem": statistics.fmean(float(r["certa"].mean()) for r in separadas), "melhor_regra": melhor_regra, "metodos": {}}
    for m in metodos:
        quartos = {"menor": [], "maior": []}
        for r in separadas:
            ordem = np.lexsort((gerador.random(len(r["certa"])), r["notas"][m]))
            quarto = round(len(ordem) / 4)
            quartos["menor"].append(float(r["certa"][ordem[:quarto]].mean()))
            quartos["maior"].append(float(r["certa"][ordem[-quarto:]].mean()))
        diferenca = amostras[m] - amostras[melhor_regra]
        saida["metodos"][m] = {
            "auc": valores[m], "intervalo_auc": [float(v) for v in np.percentile(amostras[m], [2.5, 97.5])],
            "diferenca_para_a_melhor_regra": valores[m] - valores[melhor_regra],
            "intervalo_da_diferenca": [float(v) for v in np.percentile(diferenca, [2.5, 97.5])],
            "batem_no_quarto_de_menor_nota": statistics.fmean(quartos["menor"]),
            "batem_no_quarto_de_maior_nota": statistics.fmean(quartos["maior"])}
    do_treino = auc_medio([r["do treino"] for r in por_rodada], todos)
    saida["ligacoes_do_treino"] = {t: {"auc": do_treino[t], "com_nota_maxima": statistics.fmean(r["do treino"]["nota_maxima"][t] for r in por_rodada)}
                                   for t in KGE}

    print(f"\n== Apontar ligações suspeitas no grafo da LLM, nas ligações que o treino separou: {saida['itens_por_rodada']:.0f} "
          f"por rodada, {saida['batem']:.1%} batem com as anotações")
    print(f"   {'':20s}{'AUC':>7s}{'intervalo de 95%':>18s}{'diferença para a melhor regra':>32s}{'batem: 25% de menor nota':>27s}"
          f"{'25% de maior nota':>19s}")
    for m in sorted(metodos, key=lambda m: -valores[m]):
        x = saida["metodos"][m]
        intervalo = f"{x['intervalo_auc'][0]:.3f} a {x['intervalo_auc'][1]:.3f}"
        diferenca = ("(é ela)" if m == melhor_regra else
                     f"{x['diferenca_para_a_melhor_regra']:+.3f} ({x['intervalo_da_diferenca'][0]:+.3f} a {x['intervalo_da_diferenca'][1]:+.3f})")
        print(f"   {m:20s}{x['auc']:7.3f}{intervalo:>18s}{diferenca:>32s}{x['batem_no_quarto_de_menor_nota']:27.1%}"
              f"{x['batem_no_quarto_de_maior_nota']:19.1%}")
    print("   Nas ligações com que o modelo treinou, a nota não separa nada, porque ele as decorou:")
    for t, x in saida["ligacoes_do_treino"].items():
        print(f"   {t:20s}{x['auc']:7.3f}   nota máxima em {x['com_nota_maxima']:.0%} das ligações")
    return saida


# ---------------------------------------------------------------- as partes que usam os dois grafos
def nos_dois_grafos(parte, rodadas):
    notas = sugestao.ler_notas()
    saida = {}
    for base in BASES:
        por_nota = (sugestao.triplas_do_texto(notas, atualizacao.GRAFO_DA_LLM) if base == "llm"
                    else sugestao.triplas_das_anotacoes(notas))
        medidas, tamanhos = [], []
        for k in rodadas:
            inicio = time.time()
            triplas, entidades, indice, vizinhos = grafo_da_rodada(k, notas, por_nota)
            if parte == "categoria":
                m, t = categoria_na_rodada(base, k, triplas, entidades, indice, vizinhos)
            else:
                m, t = duplicatas_na_rodada(base, k, entidades, vizinhos)
            medidas.append(m)
            tamanhos.append(t)
            print(f"   {BASES[base]['nome']}, rodada {k}: {t} ({time.time() - inicio:.0f} s)", flush=True)
        saida[base] = {"tamanhos": resumir_rodadas(tamanhos), "metodos": resumir_rodadas(medidas)}
    return saida


def mostrar_categoria(dados):
    for base, d in dados.items():
        t = d["tamanhos"]
        print(f"\n== Prever o grupo do conceito, {BASES[base]['nome']}: {t['conceitos']['media']:.0f} conceitos com grupo, "
              f"{t['grupos']['media']:.0f} grupos (média das rodadas)")
        faixas = [f[0] for f in FAIXAS_DE_GRAU]

        def melhor(m):
            """O classificador com mais acertos para esta entrada."""
            return max(d["metodos"][m], key=lambda c: d["metodos"][m][c]["acertos"]["media"])

        print(f"   {'':22s}" + "".join(f"{f'acertos, {c}':>28s}" for c in CLASSIFICADORES)
              + f"{'no melhor dos dois: desvio':>28s}{'F1 médio':>10s}" + "".join(f"{f:>20s}" for f in faixas))
        for m in sorted(d["metodos"], key=lambda m: -d["metodos"][m][melhor(m)]["acertos"]["media"]):
            x = d["metodos"][m][melhor(m)]
            colunas = [f"{d['metodos'][m][c]['acertos']['media']:.1%}" if c in d["metodos"][m] else
                       f"{x['acertos']['media']:.1%}" for c in CLASSIFICADORES]
            print(f"   {m:22s}" + "".join(f"{c:>28s}" for c in colunas) + f"{x['acertos']['desvio']:28.1%}{x['f1']['media']:10.3f}"
                  + "".join(f"{x[f]['media']:20.1%}" for f in faixas))


def mostrar_duplicatas(dados):
    for base, d in dados.items():
        t = d["tamanhos"]
        print(f"\n== Achar a gêmea, {BASES[base]['nome']}: {t['consultas']['media']:.0f} entidades com gêmea entre "
              f"{t['entidades']['media']:.0f}; {t['consultas com 2 vizinhos ou mais']['media']:.0f} delas com 2 vizinhos ou mais")
        colunas = [f"entre as {p}" for p in POSICOES] + [f"entre as {p}, com 2 vizinhos ou mais" for p in POSICOES]
        print(f"   {'':20s}" + "".join(f"{c:>38s}" for c in colunas))
        for m in sorted(d["metodos"], key=lambda m: -d["metodos"][m][colunas[1]]["media"]):
            print(f"   {m:20s}" + "".join(f"{d['metodos'][m][c]['media']:38.1%}" for c in colunas))


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("categoria", "duplicatas", "erros", "tudo"))
    argumentos.add_argument("--rodadas", type=int, default=5, help="quantas das 5 rodadas usar (padrão: todas)")
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)
    rodadas = range(1, args.rodadas + 1)

    completa = args.rodadas == 5
    saida = SAIDA_JSON if completa else SAIDA_JSON.replace(".json", "_teste.json")
    dados = sugestao.ler_json(saida) if os.path.isfile(saida) else {}
    for parte in (("categoria", "duplicatas", "erros") if args.parte == "tudo" else (args.parte,)):
        print(f"\n######## {parte}")
        if parte == "erros":
            dados[parte] = parte_erros(rodadas)
        else:
            dados[parte] = nos_dois_grafos(parte, rodadas)
            (mostrar_categoria if parte == "categoria" else mostrar_duplicatas)(dados[parte])
        retreino.salvar_json(dados, saida)
    print(f"\nResultados em {os.path.relpath(saida, RAIZ)}")


if __name__ == "__main__":
    main()
