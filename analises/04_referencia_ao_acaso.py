"""
Valor de referência "ao acaso" para a predição de links.

Calcula o MRR e o Hits@K esperados de um modelo que ordenasse as entidades
candidatas ao acaso, e compara com os resultados dos modelos em
results/link_prediction_resultados.json (gerado por compare_models.py).

Números citados no cap. 3 (métricas de predição de links) e no cap. 4
(tabela e interpretação da predição de links).

Como é o cálculo: se a entidade correta cai numa posição sorteada entre
1 e N, cada posição tem chance 1/N. Então:
  - o MRR esperado é a média de 1/1, 1/2, ..., 1/N;
  - o Hits@K esperado é K/N (chance de cair entre as K primeiras).
N é o número de entidades candidatas: as que têm ao menos uma relação,
porque só elas entram no treinamento do PyKEEN.

Uso (da raiz do repositório):
    python analises/04_referencia_ao_acaso.py [pasta_com_os_xml]
"""
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "raw")
RESULTADOS = os.path.join(RAIZ, "results", "link_prediction_resultados.json")


def chave(texto, tag):
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def entidades_com_relacao(pasta):
    entidades = set()
    for arquivo in sorted(glob.glob(os.path.join(pasta, "*.xml"))):
        raiz = ET.parse(arquivo).getroot()
        local = {a.get("id"): chave(a.get("text"), a.get("tag")) for a in raiz.findall(".//annotation")}
        for rel in raiz.findall(".//rel"):
            origem = local.get(rel.get("annotation1"))
            destino = local.get(rel.get("annotation2"))
            if origem and destino:
                entidades.update((origem, destino))
    return len(entidades)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n = entidades_com_relacao(PASTA_XML)
    mrr = sum(1 / k for k in range(1, n + 1)) / n
    hits = {k: k / n for k in (1, 3, 10)}

    print(f"1. Entidades candidatas (com ao menos uma relação): {n}  [texto: 7.961]")
    print("\n2. Valores esperados ao acaso  [texto: MRR 0,0012; Hits@1 0,0001; Hits@3 0,0004; Hits@10 0,0013]")
    print(f"   MRR:     {mrr:.4f}  ({mrr:.6f})")
    for k, v in hits.items():
        print(f"   Hits@{k:<2d}: {v:.4f}  ({v:.6f})")

    if not os.path.exists(RESULTADOS):
        print(f"\n(arquivo {RESULTADOS} não encontrado: rode compare_models.py para comparar com os modelos)")
        return
    resumo = json.load(open(RESULTADOS, encoding="utf-8"))["resumo"]
    print("\n3. Modelos comparados com o acaso")
    print("   [texto: RotatE cerca de 30 vezes acima no MRR; TransE cerca de 4 vezes;")
    print("    ComplEx a menos de um desvio padrão do acaso no MRR e no Hits@10]")
    for linha in resumo:
        m, dp = linha["MRR_media"], linha["MRR_desvio"]
        h10, dp10 = linha["Hits@10_media"], linha["Hits@10_desvio"]
        print(f"   {linha['Modelo']:8s} MRR {m:.4f} ± {dp:.4f} -> {m / mrr:5.1f} vezes o acaso;"
              f" diferença para o acaso {'menor' if abs(m - mrr) < dp else 'maior'} que 1 desvio padrão")
        print(f"   {'':8s} Hits@10 {h10:.4f} ± {dp10:.4f} -> {h10 / hits[10]:5.1f} vezes o acaso;"
              f" diferença para o acaso {'menor' if abs(h10 - hits[10]) < dp10 else 'maior'} que 1 desvio padrão")


if __name__ == "__main__":
    main()
