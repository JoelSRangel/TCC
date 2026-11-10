"""
Experimento 8: atualizar o grafo com prontuários novos.

O grafo serve para a busca, e os modelos de vetores ajudam a decidir o que entra nele. A clínica tem um grafo inicial,
revisado por pessoas; quando chega um prontuário novo, o BioBERTpt marca os conceitos, a LLM propõe as ligações e o
modelo de vetores dá uma nota a cada ligação, pelo que o grafo já sabe. Nota alta entra no grafo, e nota baixa vai para
revisão. O modelo também sugere ligações que a LLM não fez.

Simulação. As notas são divididas nas mesmas 5 partes do experimento 5d. O grafo inicial são as anotações dos
especialistas em 4 partes (a chave dos conceitos é a do experimento 1), e as notas da 5ª parte chegam como prontuários
novos; depois as partes trocam de lugar. As ligações propostas são as do grafo completo do 5g para aquela nota, e o
gabarito são as anotações da própria nota, comparadas como no experimento 5 (nos dois sentidos e aceitando textos que
casam em parte). As 20 notas do piloto e as 2 de exemplo do 5g ficam no grafo inicial, mas não contam na medida.

Cada ligação proposta cai em um de três grupos, pelo texto dos dois conceitos:
  já no grafo     os dois conceitos já estão ligados no grafo inicial
  avaliável       os dois conceitos estão no grafo inicial, mas não ligados entre si: o modelo dá uma nota
  conceito novo   algum dos dois não está no grafo inicial: o modelo não tem como avaliar
Só o grupo avaliável passa pelo filtro; os outros dois ficam como estão.

A nota de uma ligação é o percentil do conceito de destino entre todos os conceitos que o modelo ordena para a origem:
1 é o mais provável de todos. Vale o melhor dos dois sentidos, e, quando um texto corresponde a mais de uma entidade (o
mesmo texto com categorias diferentes), a melhor combinação. A ligação "nega" da LLM é avaliada como negation_of, e as
outras, como associated_with.

Técnicas: TransE, ComplEx e RotatE com a configuração escolhida no experimento 1 (grade ampliada), e Node2Vec com os
parâmetros de src/embeddings.py (a implementação do experimento 7), que usa o cosseno e não distingue os tipos de
ligação. Duas regras simples, sem vetores, servem de comparação: o número de vizinhos em comum dos dois conceitos e o
produto dos números de vizinhos (popularidade).

Medidas:
  filtro       AUC (a chance de uma ligação certa ter nota maior que uma errada; 0,5 é o acaso); e, mantendo só as 25%,
               50% ou 75% de maior nota, quantas batem com as anotações e quantas das certas ficam
  fluxo        as ligações confirmadas e as ligações anotadas encontradas no fluxo inteiro, antes e depois do filtro
  complemento  as 5 sugestões de maior nota por nota, entre pares de conceitos conhecidos que a LLM não ligou e que o
               grafo inicial não liga: quantas batem com as anotações, comparadas com sugerir ao acaso

O grafo inicial pode ser outro: com --base llm, ele é o grafo que a LLM montou (o completo do 5g) nas 4 partes, os modelos
usam a configuração escolhida no experimento 6 e cada ligação é avaliada com o seu próprio tipo. As ligações propostas e o
gabarito são os mesmos. Como os grupos dependem do grafo inicial, comparar também mede as duas bases nas ligações que são
avaliáveis nas duas, com o intervalo de 95% da diferença.

Saídas (não versionadas):
  results/experimento8/modelos/          os modelos treinados em cada parte, inteiros, com os mapas de conceitos
  results/experimento8/                  números por nota e parte, sem texto das notas (base_llm/: o mesmo, com o grafo da LLM)
  results/experimento8_resultados.json   a tabela final (experimento8_base_llm_resultados.json: com o grafo da LLM)
  results/experimento8_comparacao_das_bases.json   as duas bases nas mesmas ligações

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/08_atualizacao_do_grafo.py avaliar --rapido   # testa o código numa parte, com 20 épocas
    python experimentos/08_atualizacao_do_grafo.py avaliar            # continua de onde parou, se for interrompido
    python experimentos/08_atualizacao_do_grafo.py avaliar --base llm # o mesmo, com o grafo da LLM como grafo inicial
    python experimentos/08_atualizacao_do_grafo.py comparar           # as tabelas finais, com o que já foi salvo
"""
import argparse
import collections
import functools
import glob
import importlib.util
import itertools
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pykeen
import torch
from scipy.stats import rankdata

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento8")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento8_resultados.json")
SAIDA_COMPARACAO = os.path.join(RAIZ, "results", "experimento8_comparacao_das_bases.json")
GRAFO_DA_LLM = "5g_completo"
# O grafo inicial: de onde vêm as triplas e a configuração dos modelos, e os tipos de ligação que ele tem
BASES = {
    "anotacoes": {"nome": "grafo das anotações", "grafo": "anotacoes", "associacao": "associated_with",
                  "negacao": "negation_of", "pasta": PASTA, "saida": SAIDA_JSON},
    "llm": {"nome": "grafo da LLM", "grafo": GRAFO_DA_LLM, "associacao": "associado_a", "negacao": "nega",
            "pasta": os.path.join(PASTA, "base_llm"),
            "saida": os.path.join(RAIZ, "results", "experimento8_base_llm_resultados.json")},
}
TECNICAS = ("TransE", "ComplEx", "RotatE", "Node2Vec")
REGRAS = ("vizinhos em comum", "popularidade")
CORTES = (0.25, 0.5, 0.75)
SUGESTOES_POR_NOTA = 5
REAMOSTRAS = 2000


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Notas, triplas e treino: os do experimento 7; comparação de textos e gabarito: os dos experimentos 5
sugestao = carregar("sugestao", "07_sugestao_de_diagnosticos.py")
retreino, g5, casa, texto_da = sugestao.retreino, sugestao.g5, sugestao.casa, sugestao.texto_da
piloto, grafo, ligacoes = g5.piloto, g5.grafo, g5.ligacoes


# ---------------------------------------------------------------- a chegada dos prontuários
def pares_anotados(caminho):
    """Os pares de conceitos ligados nas anotações da nota, sem direção e sem repetir."""
    pares, vistos = [], set()
    for a, _, b in ligacoes.relacoes_entre_conceitos(caminho):
        x, y = piloto.normalizar(a), piloto.normalizar(b)
        if x and y and x != y and frozenset((x, y)) not in vistos:
            vistos.add(frozenset((x, y)))
            pares.append((x, y))
    return pares


def combina(a, b, gabarito):
    """Os índices dos pares do gabarito que a ligação entre a e b encontra."""
    return [i for i, (x, y) in enumerate(gabarito) if (casa(a, x) and casa(b, y)) or (casa(a, y) and casa(b, x))]


def tipo_no_grafo(relacao, base):
    """O tipo com que a ligação proposta é avaliada: o dela, no grafo da LLM; nas anotações, associação ou negação."""
    if base == "llm":
        return relacao
    return BASES[base]["negacao"] if relacao == "nega" else BASES[base]["associacao"]


def preparar_parte(k, notas, por_nota, propostas, fora, base):
    """O grafo inicial da parte k e, para cada nota nova, as ligações propostas e os pares candidatos a complemento.
    Em "conteudo" ficam os textos, que só servem para calcular as notas e não são gravados."""
    triplas = set().union(*(t for a, t in por_nota.items() if notas[a]["parte"] != k))
    ligados, vizinhos = set(), collections.defaultdict(set)
    for o, _, d in triplas:
        a, b = texto_da(o), texto_da(d)
        ligados |= {(a, b), (b, a)}
        vizinhos[a].add(b)
        vizinhos[b].add(a)

    numeros, conteudo = {}, {}
    for arquivo, n in notas.items():
        if n["parte"] != k or arquivo in fora:
            continue
        gabarito = pares_anotados(n["caminho"])
        links, textos = [], []
        for o_chave, relacao, d_chave in sorted(propostas[arquivo]):
            o, d = texto_da(o_chave), texto_da(d_chave)
            grupo = ("ja no grafo" if (o, d) in ligados else
                     "avaliavel" if o in vizinhos and d in vizinhos else "conceito novo")
            achados = combina(o, d, gabarito)
            links.append({"grupo": grupo, "certa": bool(achados), "achados": achados})
            textos.append((o, d, tipo_no_grafo(relacao, base)))

        conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in n["lista"]
                     if c["categoria"] not in ligacoes.LITERAIS and piloto.normalizar(c["texto"])}
        candidatos, pares = [], []
        for a, b in itertools.combinations(sorted(conceitos), 2):
            if a in vizinhos and b in vizinhos and (a, b) not in ligados and not any(
                    (casa(a, o) and casa(b, d)) or (casa(a, d) and casa(b, o)) for o, d, _ in textos):
                candidatos.append({"achados": combina(a, b, gabarito)})
                pares.append((a, b, BASES[base]["negacao" if "negacao" in (conceitos[a], conceitos[b]) else "associacao"]))
        numeros[arquivo] = {"gabarito": len(gabarito), "links": links, "candidatos": candidatos}
        conteudo[arquivo] = {"links": textos, "candidatos": pares}
    return triplas, vizinhos, numeros, conteudo


# ---------------------------------------------------------------- as notas de cada técnica
def percentil(pontos):
    """A posição de cada conceito na ordem do modelo, de 0 (o último) a 1 (o primeiro)."""
    return rankdata(pontos, method="average").astype(np.float32) / len(pontos)


def por_texto(entidade_para_id):
    ids = collections.defaultdict(list)
    for e, i in entidade_para_id.items():
        ids[texto_da(e)].append(i)
    return ids


def nota_de_par(ids_de, percentis):
    """A nota de um par de conceitos: o melhor percentil, nos dois sentidos, entre as entidades de cada texto."""
    def nota(a, b, relacao):
        return float(max(max(percentis(x, relacao)[y], percentis(y, relacao)[x]) for x in ids_de[a] for y in ids_de[b]))
    return nota


def nota_kge(modelo, fabrica):
    modelo.eval()
    relacoes = fabrica.relation_to_id

    @functools.lru_cache(maxsize=None)
    def percentis(e, relacao):
        with torch.no_grad():
            return percentil(modelo.score_t(torch.tensor([[e, relacoes[relacao]]]))[0].numpy())
    return nota_de_par(por_texto(fabrica.entity_to_id), percentis)


def nota_node2vec(indice, vetores):
    @functools.lru_cache(maxsize=None)
    def percentis(e, _):
        return percentil(vetores @ vetores[e])
    return nota_de_par(por_texto(indice), percentis)


def notas_das_regras(vizinhos):
    return {"vizinhos em comum": lambda a, b, _: float(len(vizinhos[a] & vizinhos[b])),
            "popularidade": lambda a, b, _: float(len(vizinhos[a]) * len(vizinhos[b]))}


def dar_notas(nota, conteudo, numeros):
    """As notas das ligações avaliáveis e dos candidatos a complemento de cada nota."""
    saida = {}
    for arquivo, textos in conteudo.items():
        links = [nota(*t) if numeros[arquivo]["links"][i]["grupo"] == "avaliavel" else None
                 for i, t in enumerate(textos["links"])]
        saida[arquivo] = {"links": links, "candidatos": [nota(*t) for t in textos["candidatos"]]}
    return saida


def caminho(pasta, nome, k):
    return os.path.join(pasta, f"{nome.replace(' ', '_')}_parte{k}.json")


def avaliar(args):
    configuracao = BASES[args.base]
    pasta = os.path.join(configuracao["pasta"], "rapido") if args.rapido else configuracao["pasta"]
    modelos = os.path.join(pasta, "modelos")
    os.makedirs(modelos, exist_ok=True)
    notas = sugestao.ler_notas()
    fora = set(sugestao.ler_json(os.path.join(RAIZ, "results", "experimento5g_resultados.json"))["fora_das_medidas"])
    propostas = sugestao.triplas_do_texto(notas, GRAFO_DA_LLM)
    por_nota = propostas if args.base == "llm" else sugestao.triplas_das_anotacoes(notas)
    print(f"Grafo inicial: {configuracao['nome']}")

    for k in ([1] if args.rapido else range(1, grafo.PARTES + 1)):
        print(f"\n== Parte {k} como prontuários novos", flush=True)
        triplas, vizinhos, numeros, conteudo = preparar_parte(k, notas, por_nota, propostas, fora, args.base)
        grupos = collections.Counter(l["grupo"] for n in numeros.values() for l in n["links"])
        print(f"   {len(triplas)} ligações no grafo inicial; {len(numeros)} notas novas; propostas: {dict(grupos)};"
              f" candidatos a complemento: {sum(len(n['candidatos']) for n in numeros.values())}")
        retreino.salvar_json({"parte": k, "notas": numeros}, caminho(pasta, "base", k))

        for regra, nota in notas_das_regras(vizinhos).items():
            retreino.salvar_json({"tecnica": regra, "parte": k, "notas": dar_notas(nota, conteudo, numeros)},
                                 caminho(pasta, regra, k))

        for tecnica in TECNICAS:
            arquivo = caminho(pasta, tecnica, k)
            if os.path.exists(arquivo):
                continue
            inicio = time.time()
            if tecnica == "Node2Vec":
                indice, vetores = sugestao.treinar_node2vec(triplas, k, 1 if args.rapido else sugestao.N2V_EPOCAS)
                np.savez_compressed(os.path.join(modelos, f"Node2Vec_parte{k}.npz"), ids=np.array(list(indice)), vetores=vetores)
                nota, info = nota_node2vec(indice, vetores), {}
            else:
                modelo, fabrica, info = sugestao.treinar_kge(triplas, configuracao["grafo"], tecnica, k,
                                                             20 if args.rapido else retreino.MAX_EPOCAS, 1 if args.rapido else 3)
                torch.save(modelo, os.path.join(modelos, f"{tecnica}_parte{k}.pt"))
                retreino.salvar_json({"entidades": fabrica.entity_to_id, "relacoes": fabrica.relation_to_id},
                                     os.path.join(modelos, f"{tecnica}_parte{k}_mapas.json"))
                nota = nota_kge(modelo, fabrica)
            resultado = dar_notas(nota, conteudo, numeros)
            retreino.salvar_json({"tecnica": tecnica, "parte": k, "info": info, "notas": resultado}, arquivo)
            print(f"   {tecnica:10s} {time.time() - inicio:7.1f} s", flush=True)
    comparar(pasta, os.path.join(pasta, "resultados.json") if args.rapido else configuracao["saida"])


# ---------------------------------------------------------------- a tabela
def auc(certas, notas):
    postos = rankdata(notas)
    n1 = certas.sum()
    return (postos[certas].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(certas) - n1))


def manter(notas, fracao, gerador):
    """Os índices das ligações mantidas: a fração de maior nota, com os empates sorteados."""
    ordem = np.lexsort((gerador.random(len(notas)), -notas))
    return ordem[:round(fracao * len(notas))]


def carregar_resultados(pasta):
    base, por_tecnica = {}, collections.defaultdict(dict)
    for arquivo in sorted(glob.glob(os.path.join(pasta, "*_parte*.json"))):
        d = sugestao.ler_json(arquivo)
        if "tecnica" in d:
            por_tecnica[d["tecnica"]].update(d["notas"])
        else:
            base.update(d["notas"])
    return base, por_tecnica


def comparar(pasta, destino):
    base, por_tecnica = carregar_resultados(pasta)
    notas = sorted(base)
    gerador = np.random.default_rng(0)

    # Índices das ligações avaliáveis, por nota, em ordem
    certas, de_nota, posicao = [], [], []
    grupos = collections.Counter()
    certas_grupo = collections.Counter()
    for i, a in enumerate(notas):
        for j, l in enumerate(base[a]["links"]):
            grupos[l["grupo"]] += 1
            certas_grupo[l["grupo"]] += l["certa"]
            if l["grupo"] == "avaliavel":
                certas.append(l["certa"])
                de_nota.append(i)
                posicao.append((a, j))
    certas, de_nota = np.array(certas), np.array(de_nota)
    total = sum(grupos.values())
    print(f"\n== {len(notas)} prontuários novos, {total} ligações propostas")
    for g in ("ja no grafo", "avaliavel", "conceito novo"):
        print(f"   {g:14s} {grupos[g] / total:6.1%} das ligações; {certas_grupo[g] / grupos[g]:6.1%} batem com as anotações")

    def fluxo(mantidas):
        """Confirmadas e encontradas no fluxo inteiro, com as ligações avaliáveis mantidas (conjunto de (nota, índice))."""
        confirmadas = aceitas = encontradas = gabarito = 0
        for a in notas:
            achados = set()
            for j, l in enumerate(base[a]["links"]):
                if l["grupo"] != "avaliavel" or (a, j) in mantidas:
                    aceitas += 1
                    confirmadas += l["certa"]
                    achados.update(l["achados"])
            encontradas += len(achados)
            gabarito += base[a]["gabarito"]
        return confirmadas / aceitas, encontradas / gabarito

    todas = set(posicao)
    confirmadas_antes, encontradas_antes = fluxo(todas)
    saida = {"prontuarios": len(notas), "ligacoes": total, "grupos": {g: {"fracao": grupos[g] / total, "batem": certas_grupo[g] / grupos[g]} for g in grupos},
             "antes_do_filtro": {"confirmadas": confirmadas_antes, "encontradas": encontradas_antes}, "tecnicas": {}}
    print(f"   antes do filtro: {confirmadas_antes:.1%} confirmadas, {encontradas_antes:.1%} encontradas")

    indices_por_nota = [np.flatnonzero(de_nota == i) for i in range(len(notas))]
    sorteios = gerador.integers(0, len(notas), (REAMOSTRAS, len(notas)))
    linhas = {}
    for tecnica, resultado in por_tecnica.items():
        if set(resultado) != set(notas):
            continue
        s = np.array([resultado[a]["links"][j] for a, j in posicao], dtype=float)
        valor = auc(certas, s)
        amostras = []
        for sorteio in sorteios:
            idx = np.concatenate([indices_por_nota[i] for i in sorteio])
            amostras.append(auc(certas[idx], s[idx]))
        baixo, alto = np.percentile(amostras, [2.5, 97.5])
        r = {"auc": float(valor), "intervalo_auc": [float(baixo), float(alto)], "cortes": {}}
        for c in CORTES:
            m = manter(s, c, gerador)
            mantidas = {posicao[i] for i in m}
            confirmadas, encontradas = fluxo(mantidas)
            r["cortes"][str(c)] = {"batem": float(certas[m].mean()), "certas_mantidas": float(certas[m].sum() / certas.sum()),
                                   "fluxo_confirmadas": confirmadas, "fluxo_encontradas": encontradas}

        sugeridas = acertos = com_acerto = recuperados = faltavam = 0
        for a in notas:
            cand = base[a]["candidatos"]
            achados_llm = set().union(*(set(l["achados"]) for l in base[a]["links"])) if base[a]["links"] else set()
            faltam = set(range(base[a]["gabarito"])) - achados_llm
            faltavam += len(faltam)
            if not cand:
                continue
            ordem = np.lexsort((gerador.random(len(cand)), -np.array(resultado[a]["candidatos"], dtype=float)))[:SUGESTOES_POR_NOTA]
            achados = set().union(*(set(cand[i]["achados"]) for i in ordem))
            certos = sum(bool(cand[i]["achados"]) for i in ordem)
            sugeridas += len(ordem)
            acertos += certos
            com_acerto += certos > 0
            recuperados += len(achados & faltam)
        r["complemento"] = {"precisao": acertos / sugeridas, "notas_com_acerto": com_acerto / len(notas),
                            "faltantes_recuperadas": recuperados / faltavam}
        linhas[tecnica] = r
    saida["tecnicas"] = linhas

    candidatos = sum(len(base[a]["candidatos"]) for a in notas)
    certos_candidatos = sum(bool(c["achados"]) for a in notas for c in base[a]["candidatos"])
    saida["complemento_ao_acaso"] = certos_candidatos / candidatos if candidatos else None
    ordem = sorted(linhas, key=lambda t: -linhas[t]["auc"])
    print(f"\n== Filtro nas {len(certas)} ligações avaliáveis ({certas.mean():.1%} batem com as anotações)")
    print(f"   {'':20s}{'AUC':>7s}{'intervalo de 95%':>18s}" + "".join(f"{f'mantendo {c:.0%}':>26s}" for c in CORTES))
    print(f"   {'':45s}" + "".join(f"{'batem | certas mantidas':>26s}" for _ in CORTES))
    for t in ordem:
        r = linhas[t]
        intervalo = f"{r['intervalo_auc'][0]:.3f} a {r['intervalo_auc'][1]:.3f}"
        colunas = [f"{r['cortes'][str(c)]['batem']:.1%} | {r['cortes'][str(c)]['certas_mantidas']:.1%}" for c in CORTES]
        print(f"   {t:20s}{r['auc']:7.3f}{intervalo:>18s}" + "".join(f"{coluna:>26s}" for coluna in colunas))
    print(f"\n== Fluxo inteiro mantendo 50% das avaliáveis (antes: {confirmadas_antes:.1%} confirmadas, {encontradas_antes:.1%} encontradas)")
    for t in ordem:
        c = linhas[t]["cortes"]["0.5"]
        print(f"   {t:20s}{c['fluxo_confirmadas']:8.1%} confirmadas{c['fluxo_encontradas']:8.1%} encontradas")
    print(f"\n== Complemento: {SUGESTOES_POR_NOTA} sugestões por nota, entre {candidatos} pares (ao acaso: {saida['complemento_ao_acaso']:.1%} batem)")
    for t in sorted(linhas, key=lambda t: -linhas[t]["complemento"]["precisao"]):
        c = linhas[t]["complemento"]
        print(f"   {t:20s}{c['precisao']:7.1%} batem | notas com acerto {c['notas_com_acerto']:6.1%}"
              f" | das ligações que a LLM perdeu, recuperou {c['faltantes_recuperadas']:6.1%}")
    retreino.salvar_json(saida, destino)
    print(f"\nResultados em {os.path.relpath(destino, RAIZ)}")


def comparar_bases():
    """As duas bases nas ligações avaliáveis nas duas: a mesma lista de ligações, julgada por modelos treinados em cada
    grafo."""
    (base_a, tecnicas_a), (base_l, tecnicas_l) = (carregar_resultados(BASES[b]["pasta"]) for b in ("anotacoes", "llm"))
    notas = sorted(set(base_a) & set(base_l))
    posicoes = []
    for a in notas:
        assert len(base_a[a]["links"]) == len(base_l[a]["links"]), "as ligações propostas deveriam ser as mesmas"
        posicoes += [(a, j) for j, (x, y) in enumerate(zip(base_a[a]["links"], base_l[a]["links"]))
                     if x["grupo"] == y["grupo"] == "avaliavel"]
    certas = np.array([base_a[a]["links"][j]["certa"] for a, j in posicoes])
    indice_da_nota = {a: i for i, a in enumerate(notas)}
    de_nota = np.array([indice_da_nota[a] for a, _ in posicoes])
    indices = [np.flatnonzero(de_nota == i) for i in range(len(notas))]
    sorteios = np.random.default_rng(0).integers(0, len(notas), (REAMOSTRAS, len(notas)))
    print(f"\n== As duas bases nas mesmas {len(posicoes)} ligações ({certas.mean():.1%} batem com as anotações)")
    print(f"   {'':20s}{'AUC anotações':>15s}{'AUC grafo da LLM':>18s}{'diferença':>11s}{'intervalo de 95%':>20s}")
    saida = {"ligacoes": len(posicoes), "batem": float(certas.mean()), "tecnicas": {}}
    for t in [*TECNICAS, *REGRAS]:
        s_a = np.array([tecnicas_a[t][a]["links"][j] for a, j in posicoes], dtype=float)
        s_l = np.array([tecnicas_l[t][a]["links"][j] for a, j in posicoes], dtype=float)
        diferencas = []
        for sorteio in sorteios:
            idx = np.concatenate([indices[i] for i in sorteio])
            diferencas.append(auc(certas[idx], s_a[idx]) - auc(certas[idx], s_l[idx]))
        baixo, alto = np.percentile(diferencas, [2.5, 97.5])
        saida["tecnicas"][t] = {"auc_anotacoes": float(auc(certas, s_a)), "auc_llm": float(auc(certas, s_l)),
                                "diferenca": float(auc(certas, s_a) - auc(certas, s_l)), "intervalo": [float(baixo), float(alto)]}
        r = saida["tecnicas"][t]
        intervalo = f"{baixo:+.3f} a {alto:+.3f}"
        print(f"   {t:20s}{r['auc_anotacoes']:15.3f}{r['auc_llm']:18.3f}{r['diferenca']:+11.3f}{intervalo:>20s}")
    print("   (diferença positiva: treinar nas anotações separa melhor)")
    retreino.salvar_json(saida, SAIDA_COMPARACAO)
    print(f"\nResultados em {os.path.relpath(SAIDA_COMPARACAO, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    partes = argumentos.add_subparsers(dest="parte", required=True)
    a = partes.add_parser("avaliar", help="os prontuários de cada parte chegando ao grafo inicial")
    a.add_argument("--rapido", action="store_true", help="uma parte só, com 20 épocas: testa o código")
    a.add_argument("--base", choices=list(BASES), default="anotacoes", help="o grafo inicial (padrão: o das anotações)")
    partes.add_parser("comparar", help="as tabelas finais, com o que já foi salvo")
    args = argumentos.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)
    print(f"PyKEEN {pykeen.get_version()} | PyTorch {torch.__version__} | {args.threads} núcleos")
    if args.parte == "avaliar":
        avaliar(args)
    else:
        for b in BASES:
            print(f"\n######## {BASES[b]['nome']}")
            comparar(BASES[b]["pasta"], BASES[b]["saida"])
        comparar_bases()


if __name__ == "__main__":
    main()
