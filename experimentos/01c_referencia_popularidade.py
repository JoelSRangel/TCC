"""
Experimento 1c: os modelos superam uma referência de popularidade?

O acaso (MRR de 0,0012) é uma referência fraca: neste grafo, poucas
entidades concentram muitas ligações. Uma regra simples, sem nenhum
treino, como "para prever o destino de uma relação, ordene as entidades
pelo número de vezes que já foram destino dessa relação no treino", já
pode acertar muito mais.

Este script avalia essa referência de popularidade, pronta no PyKEEN
(MarginalDistributionBaseline, só com a frequência por relação): para
cada tripla do teste, ordena as entidades pelo número de vezes que cada
uma ocupou aquela posição (origem ou destino) daquela relação no treino.
A divisão das triplas e a avaliação são as mesmas do experimento 1 (teste
filtrado pelas triplas de treino e de validação; no total, por tipo de
relação e por lado).

Não há treino nem sementes: o resultado é determinístico. No fim, mostra
lado a lado os modelos treinados no experimento 1 (grade ampliada).

Saída (não versionada):
  results/experimento1c_popularidade.json

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/01c_referencia_popularidade.py
"""
import importlib.util
import json
import logging
import os
import sys
import warnings

import pykeen
import torch
from pykeen.models import MarginalDistributionBaseline
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento1c_popularidade.json")
RESULTADOS_TREINADOS = os.path.join(RAIZ, "results", "experimento1_ampliado_resultados.json")

# Leitura das triplas, divisão e avaliação: as mesmas funções do experimento 1
_spec = importlib.util.spec_from_file_location("retreino", os.path.join(RAIZ, "experimentos", "01_retreino_kge.py"))
retreino = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retreino)

REFERENCIAS = {"popularidade": dict(entity_margin=False, relation_margin=True)}


def linha(nome, m):
    return (f"   {nome:22s} MRR {m['MRR']:.4f} | Hits@1 {m['Hits@1']:.4f} | Hits@10 {m['Hits@10']:.4f}"
            f" | AMRI {m['AMRI']:.4f}")


def avaliar_referencias(treino, validacao, teste):
    """Cada referência de REFERENCIAS no teste, avaliada como os modelos do experimento 1 (o 6 usa
    a mesma função nos grafos do 5g)."""
    resultados = {}
    for nome, parametros in REFERENCIAS.items():
        modelo = MarginalDistributionBaseline(triples_factory=treino, **parametros)
        # O avaliador do PyKEEN 1.11 descobre o dispositivo pelos parâmetros ou buffers do modelo,
        # e as referências não têm nenhum: um buffer vazio resolve, sem mudar os escores.
        modelo.register_buffer("dispositivo", torch.empty(0))
        total, por_relacao = retreino.avaliar_teste(modelo, treino, validacao, teste)
        resultados[nome] = {"teste": total, "teste_por_relacao": por_relacao}
    return resultados


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)

    fabrica = TriplesFactory.from_labeled_triples(triples=retreino.ler_triplas(retreino.PASTA_XML))
    treino, validacao, teste = fabrica.split(list(retreino.DIVISAO), random_state=retreino.SEMENTE_DIVISAO)
    print(f"PyKEEN {pykeen.get_version()} | {treino.num_triples} treino, {validacao.num_triples} validação,"
          f" {teste.num_triples} teste | entidades: {treino.num_entities}")

    dados = {"referencias": REFERENCIAS, "referencia_ao_acaso": retreino.referencia_ao_acaso(treino.num_entities),
             "resultados": avaliar_referencias(treino, validacao, teste)}
    retreino.salvar_json(dados, SAIDA_JSON)

    print("\n== Teste, todas as triplas")
    acaso = dados["referencia_ao_acaso"]
    print(linha("acaso", acaso))
    for nome, r in dados["resultados"].items():
        print(linha(nome, r["teste"]))
    treinados = {}
    if os.path.isfile(RESULTADOS_TREINADOS):
        with open(RESULTADOS_TREINADOS, encoding="utf-8") as f:
            treinados = json.load(f)["resumo"]
        for rotulo, r in treinados.items():
            print(linha(f"{rotulo} (média)", {m: r["teste"][m]["media"] for m in retreino.METRICAS}))

    print("\n== Por tipo de relação e lado (MRR ao prever a origem | ao prever o destino)")
    for relacao in sorted(dados["resultados"]["popularidade"]["teste_por_relacao"]):
        triplas = dados["resultados"]["popularidade"]["teste_por_relacao"][relacao]["triplas"]
        print(f"   {relacao} ({triplas} triplas no teste)")
        for nome, r in dados["resultados"].items():
            m = r["teste_por_relacao"][relacao]
            print(f"      {nome:22s} MRR {m['MRR']:.4f} (origem {m['MRR_origem']:.4f} | destino {m['MRR_destino']:.4f})")
        for rotulo, r in treinados.items():
            m = r["teste_por_relacao"][relacao]
            print(f"      {rotulo + ' (média)':22s} MRR {m['MRR']['media']:.4f}"
                  f" (origem {m['MRR_origem']['media']:.4f} | destino {m['MRR_destino']['media']:.4f})")
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


if __name__ == "__main__":
    main()
