"""
Experimento 3b: o que a entidade e seu vizinho mais próximo têm em comum no grafo?

No experimento 3, a similaridade de cosseno de cada entidade com a mais
parecida ficou muito acima da de vetores aleatórios. Similaridade alta não
quer dizer significado parecido: os modelos só veem as ligações do grafo,
e duas entidades com exatamente as mesmas ligações (por exemplo, dois
conceitos diferentes que só aparecem negados pelo mesmo "SEM") tendem a
receber quase o mesmo vetor, tenham ou não algo em comum clinicamente.

Para cada modelo e semente do experimento 1 (grade ampliada), este script
classifica cada par (entidade, vizinho mais próximo) pelo grafo de treino:
  equivalentes      as duas têm exatamente as mesmas ligações (mesmas
                    entidades, mesmas relações, mesma direção)
  ligadas entre si  uma tripla de treino liga uma à outra
  ligação em comum  não são equivalentes, mas estão ligadas a pelo menos
                    uma mesma entidade
  outras            nenhum dos casos acima
e mostra a fração de entidades e a mediana da similaridade em cada grupo.
A preparação dos vetores é a mesma do experimento 3.

Saída (não versionada):
  results/experimento3b_vizinhos_estrutura.json

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/03b_vizinhos_estrutura.py
"""
import collections
import glob
import importlib.util
import os
import re
import statistics
import sys

import numpy as np
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_VETORES = os.path.join(RAIZ, "results", "experimento1_ampliado_vetores")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento3b_vizinhos_estrutura.json")
GRUPOS = ("equivalentes", "ligadas entre si", "ligação em comum", "outras")


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Triplas, divisão e preparação dos vetores: as mesmas dos experimentos 1 e 3
retreino = carregar("retreino", "01_retreino_kge.py")
agrupamento = carregar("agrupamento", "03_agrupamento.py")


def estrutura(treino):
    """Ligações (relação, direção, outra entidade) e entidades vizinhas de cada entidade no treino."""
    ligacoes, vizinhos = collections.defaultdict(set), collections.defaultdict(set)
    for h, r, t in treino.mapped_triples.tolist():
        ligacoes[h].add((r, "sai", t))
        ligacoes[t].add((r, "entra", h))
        vizinhos[h].add(t)
        vizinhos[t].add(h)
    return {e: frozenset(s) for e, s in ligacoes.items()}, vizinhos


def mais_proximo(x, bloco=1000):
    """Índice e similaridade de cosseno da entidade mais parecida com cada uma (vetores já normalizados)."""
    indices, maiores = np.empty(len(x), dtype=int), np.empty(len(x))
    for inicio in range(0, len(x), bloco):
        similaridades = x[inicio:inicio + bloco] @ x.T
        linhas = np.arange(similaridades.shape[0])
        similaridades[linhas, inicio + linhas] = -np.inf  # ignora a própria entidade
        indices[inicio:inicio + bloco] = similaridades.argmax(axis=1)
        maiores[inicio:inicio + bloco] = similaridades.max(axis=1)
    return indices, maiores


def grupo(a, b, ligacoes, vizinhos):
    if ligacoes[a] == ligacoes[b]:
        return "equivalentes"
    if b in vizinhos[a]:
        return "ligadas entre si"
    if vizinhos[a] & vizinhos[b]:
        return "ligação em comum"
    return "outras"


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    fabrica = TriplesFactory.from_labeled_triples(triples=retreino.ler_triplas(retreino.PASTA_XML))
    treino, _, _ = fabrica.split(list(retreino.DIVISAO), random_state=retreino.SEMENTE_DIVISAO)
    ligacoes, vizinhos = estrutura(treino)
    n = treino.num_entities
    repeticoes = collections.Counter(ligacoes.values())
    com_equivalente = sum(1 for e in range(n) if repeticoes[ligacoes[e]] > 1) / n
    uma_ligacao = sum(1 for e in range(n) if len(ligacoes[e]) == 1) / n
    print(f"Grafo de treino: {n} entidades; {100 * uma_ligacao:.1f}% com uma única ligação;"
          f" {100 * com_equivalente:.1f}% têm pelo menos uma outra entidade com exatamente as mesmas ligações")

    arquivos = sorted(glob.glob(os.path.join(PASTA_VETORES, "*_semente*.npz")))
    if not arquivos:
        sys.exit(f"Nenhum vetor encontrado em {PASTA_VETORES}. Rode antes o experimento 1 (grade ampliada).")
    dados = {"entidades": n, "fracao_uma_ligacao": uma_ligacao, "fracao_com_equivalente": com_equivalente,
             "modelos": {}}
    por_modelo = collections.defaultdict(list)
    for arquivo in arquivos:
        modelo, semente = re.match(r"(.+)_semente(\d+)\.npz$", os.path.basename(arquivo)).groups()
        conteudo = np.load(arquivo)
        entidades = np.array([treino.entity_to_id[rotulo] for rotulo in conteudo["ids"]])
        indices, maiores = mais_proximo(agrupamento.preparar(conteudo["vetores"]))
        grupos = np.array([grupo(entidades[i], entidades[j], ligacoes, vizinhos) for i, j in enumerate(indices)])
        execucao = {"semente": int(semente), "mediana": float(np.median(maiores))}
        for g in GRUPOS:
            no_grupo = maiores[grupos == g]
            execucao[g] = {"fracao": float(len(no_grupo) / n),
                           "mediana": float(np.median(no_grupo)) if len(no_grupo) else None}
        por_modelo[modelo].append(execucao)

    print("\n== A que grupo pertence o vizinho mais próximo: % das entidades (mediana da similaridade no grupo)")
    print("   média entre as sementes")
    print(f"   {'':11s}" + "".join(f"{g:>22s}" for g in GRUPOS) + f"{'todas':>10s}")
    for modelo, execucoes in por_modelo.items():
        resumo = {"sementes": len(execucoes), "mediana": statistics.fmean(e["mediana"] for e in execucoes)}
        celulas = []
        for g in GRUPOS:
            fracao = statistics.fmean(e[g]["fracao"] for e in execucoes)
            medianas = [e[g]["mediana"] for e in execucoes if e[g]["mediana"] is not None]
            mediana = statistics.fmean(medianas) if medianas else None
            resumo[g] = {"fracao": fracao, "mediana": mediana}
            celulas.append(f"{100 * fracao:5.1f}% ({'  -  ' if mediana is None else f'{mediana:.3f}'})")
        dados["modelos"][modelo] = {"resumo": resumo, "execucoes": execucoes}
        print(f"   {modelo:11s}" + "".join(f"{c:>22s}" for c in celulas) + f"{resumo['mediana']:10.3f}")
    retreino.salvar_json(dados, SAIDA_JSON)
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


if __name__ == "__main__":
    main()
