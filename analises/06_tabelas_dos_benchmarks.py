"""
Valores das tabelas de resultados do cap. 4, lidos dos JSONs dos benchmarks.

Mostra os números das tabelas de predição de links, de separação das
categorias e de K-means do jeito que aparecem no texto, para conferência:
  - results/link_prediction_resultados.json (gerado por compare_models.py)
  - results/clustering_resultados.json      (gerado por cluster_benchmark.py)

Uso (da raiz do repositório):
    python analises/06_tabelas_dos_benchmarks.py
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS = os.path.join(RAIZ, "results", "link_prediction_resultados.json")
AGRUPAMENTO = os.path.join(RAIZ, "results", "clustering_resultados.json")


def br(valor, casas):
    """Formata com vírgula decimal, como no texto."""
    return f"{valor:.{casas}f}".replace(".", ",")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    links = json.load(open(LINKS, encoding="utf-8"))
    print(f"1. Tabela de predição de links (média ± desvio, sementes {links['seeds']})")
    for linha in links["resumo"]:
        celulas = [f"{br(linha[m + '_media'], 4)} ± {br(linha[m + '_desvio'], 4)}"
                   for m in ("MRR", "Hits@1", "Hits@3", "Hits@10")]
        print(f"   {linha['Modelo']:8s} " + " | ".join(celulas))

    agrup = json.load(open(AGRUPAMENTO, encoding="utf-8"))
    print(f"\n2. Tabela de separação das categorias ({agrup['tags_distintas']} categorias)")
    for linha in agrup["resumo"]:
        print(f"   {linha['Modelo']:8s} Silhouette {br(linha['Silhouette_media'], 4)} ± {br(linha['Silhouette_desvio'], 4)}"
              f" | Davies-Bouldin {br(linha['DaviesBouldin_media'], 4)} ± {br(linha['DaviesBouldin_desvio'], 4)}")

    print(f"\n3. Tabela do K-means (K testados: {agrup['k_values']})")
    modelos = agrup["resumo"]
    print("   K   | " + " | ".join(f"{m['Modelo']:^15s}" for m in modelos))
    print("       | " + " | ".join(f"{'S':>6s}  {'DB':>6s} " for _ in modelos))
    maior_desvio = 0.0
    for k in agrup["k_values"]:
        celulas = []
        for m in modelos:
            v = m["kmeans_sweep"][str(k)]
            maior_desvio = max(maior_desvio, v["silhouette_desvio"])
            celulas.append(f"{br(v['silhouette_media'], 3):>6s}  {br(v['davies_bouldin_media'], 2):>6s} ")
        print(f"   {k:<3d} | " + " | ".join(celulas))
    print(f"   maior desvio padrão do Silhouette no K-means: {br(maior_desvio, 4)}"
          "  [legenda da tabela: no máximo 0,001]")


if __name__ == "__main__":
    main()
