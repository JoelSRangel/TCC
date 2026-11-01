"""
Experimento 3: agrupamento com os vetores dos modelos retreinados.

Usa os vetores salvos pelo experimento 1 (um arquivo por modelo e semente)
e calcula, para cada um:
  1. Separação das 85 categorias semânticas: Silhouette (distância de
     cosseno, amostra de 3.000 entidades, semente 42) e Davies-Bouldin,
     com as categorias como grupos.
  2. K-means com K = 2 a 10, 15, 20, 25, 30 e 85 (10 inicializações, semente
     igual à do modelo), avaliado pelo Silhouette e pelo Davies-Bouldin.
  3. Vizinho mais próximo: para cada entidade, a similaridade de cosseno
     com a entidade mais parecida; reporta a mediana e a fração de
     entidades com vizinho de similaridade 0,5 ou mais, ao lado do mesmo
     cálculo para vetores aleatórios do mesmo tamanho (semente 1). É a
     medida usada no protótipo de RAG.

A preparação dos vetores é a mesma de cluster_benchmark.py: nos modelos
complexos, a parte real e a imaginária são concatenadas; todos os vetores
são normalizados para comprimento 1; e as categorias são reduzidas pela
mesma regra de normalize_tag() (85 categorias). O resultado de cada modelo
é a média e o desvio padrão entre as sementes.

Saída (não versionada):
  results/experimento3_agrupamento_<origem>.json

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/03_agrupamento.py                          # vetores da grade ampliada
    python experimentos/03_agrupamento.py --origem experimento1    # vetores da grade básica
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALORES_K = (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 85)
AMOSTRA_SILHOUETTE, SEMENTE_SILHOUETTE = 3000, 42
MARCADORES = {"Abbreviation", "Negation"}


def categoria(tag_bruta):
    """Mesma regra de normalize_tag() em cluster_benchmark.py."""
    partes = []
    for parte in (tag_bruta or "").split("|"):
        if parte not in partes:
            partes.append(parte)
    substantivas = [p for p in partes if p not in MARCADORES]
    if substantivas:
        return substantivas[0]
    return partes[0] if partes else tag_bruta


def preparar(vetores):
    """Concatena parte real e imaginária (modelos complexos) e normaliza para comprimento 1."""
    if np.iscomplexobj(vetores):
        vetores = np.concatenate([vetores.real, vetores.imag], axis=1)
    vetores = vetores.astype(np.float64)
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return vetores / normas


def vizinho_mais_proximo(x, bloco=1000):
    """Maior similaridade de cosseno de cada linha com qualquer outra (vetores já normalizados)."""
    maiores = np.empty(len(x))
    for inicio in range(0, len(x), bloco):
        similaridades = x[inicio:inicio + bloco] @ x.T
        for i in range(similaridades.shape[0]):
            similaridades[i, inicio + i] = -np.inf  # ignora a própria entidade
        maiores[inicio:inicio + bloco] = similaridades.max(axis=1)
    return {"mediana": float(np.median(maiores)), "fracao_0_5_ou_mais": float(np.mean(maiores >= 0.5))}


def avaliar_vetores(x, categorias, semente):
    resultado = {
        "categorias": {
            "silhouette": float(silhouette_score(x, categorias, metric="cosine", sample_size=AMOSTRA_SILHOUETTE,
                                                 random_state=SEMENTE_SILHOUETTE)),
            "davies_bouldin": float(davies_bouldin_score(x, categorias)),
        },
        "kmeans": {},
        "vizinho": vizinho_mais_proximo(x),
    }
    for k in VALORES_K:
        rotulos = KMeans(n_clusters=k, random_state=semente, n_init=10).fit_predict(x)
        resultado["kmeans"][str(k)] = {
            "silhouette": float(silhouette_score(x, rotulos, metric="cosine", sample_size=AMOSTRA_SILHOUETTE,
                                                 random_state=SEMENTE_SILHOUETTE)),
            "davies_bouldin": float(davies_bouldin_score(x, rotulos)),
        }
    return resultado


def resumir(valores):
    return {"media": statistics.fmean(valores), "desvio": statistics.pstdev(valores) if len(valores) > 1 else 0.0}


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--origem", default="experimento1_ampliado",
                            help="prefixo da pasta de vetores em results/ (padrão: experimento1_ampliado)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    pasta = os.path.join(RAIZ, "results", f"{args.origem}_vetores")
    arquivos = sorted(glob.glob(os.path.join(pasta, "*_semente*.npz")))
    if not arquivos:
        sys.exit(f"Nenhum vetor encontrado em {pasta}. Rode antes o experimento 1.")
    por_modelo = collections.defaultdict(list)
    for arquivo in arquivos:
        modelo, semente = re.match(r"(.+)_semente(\d+)\.npz$", os.path.basename(arquivo)).groups()
        por_modelo[modelo].append((int(semente), arquivo))

    dados = {"origem": args.origem, "valores_k": VALORES_K, "modelos": {}, "aleatorio": {}}
    saida = os.path.join(RAIZ, "results", f"experimento3_agrupamento_{args.origem}.json")
    for modelo, lista in por_modelo.items():
        print(f"\n== {modelo} ({len(lista)} sementes)")
        execucoes = []
        for semente, arquivo in sorted(lista):
            inicio = time.time()
            conteudo = np.load(arquivo)
            x = preparar(conteudo["vetores"])
            categorias = np.array([categoria(entidade.split("|", 1)[1]) for entidade in conteudo["ids"]])
            r = avaliar_vetores(x, categorias, semente)
            r["semente"] = semente
            execucoes.append(r)
            dimensao = x.shape[1]
            if str(dimensao) not in dados["aleatorio"]:
                aleatorios = preparar(np.random.default_rng(1).standard_normal((len(x), dimensao)))
                dados["aleatorio"][str(dimensao)] = vizinho_mais_proximo(aleatorios)
            print(f"   semente {semente}: categorias Silhouette {r['categorias']['silhouette']:.4f},"
                  f" DB {r['categorias']['davies_bouldin']:.2f} | vizinho mediana {r['vizinho']['mediana']:.3f}"
                  f" ({time.time() - inicio:.0f} s)")
        dados["modelos"][modelo] = {
            "entidades": int(len(x)), "dimensao_real": int(x.shape[1]), "sementes": len(execucoes),
            "categorias": {m: resumir([e["categorias"][m] for e in execucoes]) for m in ("silhouette", "davies_bouldin")},
            "kmeans": {str(k): {m: resumir([e["kmeans"][str(k)][m] for e in execucoes])
                                for m in ("silhouette", "davies_bouldin")} for k in VALORES_K},
            "vizinho": {m: resumir([e["vizinho"][m] for e in execucoes]) for m in ("mediana", "fracao_0_5_ou_mais")},
            "execucoes": execucoes,
        }
        with open(saida, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    print("\n== Separação das 85 categorias (média ± desvio entre as sementes)")
    for modelo, r in dados["modelos"].items():
        s, d = r["categorias"]["silhouette"], r["categorias"]["davies_bouldin"]
        print(f"   {modelo:10s} Silhouette {s['media']:.4f} ± {s['desvio']:.4f} | Davies-Bouldin {d['media']:.2f} ± {d['desvio']:.2f}")
    print("\n== K-means: Silhouette médio por K")
    print("   " + " " * 10 + "".join(f"{k:>7d}" for k in VALORES_K))
    for modelo, r in dados["modelos"].items():
        print(f"   {modelo:10s}" + "".join(f"{r['kmeans'][str(k)]['silhouette']['media']:7.3f}" for k in VALORES_K))
    print("\n== Vizinho mais próximo (mediana da similaridade de cosseno)")
    for modelo, r in dados["modelos"].items():
        v = r["vizinho"]
        base = dados["aleatorio"][str(r["dimensao_real"])]
        print(f"   {modelo:10s} {v['mediana']['media']:.3f} ± {v['mediana']['desvio']:.3f}"
              f" (com vizinho de 0,5 ou mais: {100 * v['fracao_0_5_ou_mais']['media']:.1f}%)"
              f" | aleatórios de {r['dimensao_real']} dimensões: {base['mediana']:.3f}")
    print(f"\nResultados completos em {os.path.relpath(saida, RAIZ)}")


if __name__ == "__main__":
    main()
