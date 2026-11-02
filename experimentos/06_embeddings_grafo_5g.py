"""
Experimento 6: TransE, ComplEx e RotatE no grafo do texto (5g).

Os experimentos 1 a 4 usaram o grafo das anotações do SemClinBr, que só tem dois tipos de relação
(associated_with e negation_of). O experimento 5g montou outro grafo a partir do texto das 998 notas, com
cinco tipos: trata, tem_resultado, localizado_em, nega e associado_a. Aqui os mesmos modelos são treinados e
avaliados nesse grafo, com o mesmo código e o mesmo protocolo do experimento 1 na grade ampliada (o
modo_completo do 01_retreino_kge.py):
  1. divisão das triplas em treino, validação e teste (80/10/10, semente 42);
  2. seleção da configuração de cada modelo pela validação (semente 0, até 300 épocas, parada antecipada);
  3. avaliação final da configuração escolhida com as sementes 0 a 4, no teste filtrado, no total e por
     tipo de relação.
Só muda o grafo. A referência de popularidade é a do experimento 1c, calculada na divisão de cada grafo.

Os dois grafos do 5g (results/experimento5g/prontas/):
  completo  todas as ligações entre conceitos (grafo_completo.tsv)
  filtrado  só as ligações que respeitam as regras de categoria do experimento 5b (grafo_filtrado.tsv)

O MRR de grafos diferentes não se compara direto: o número de entidades e a concentração das ligações
mudam a dificuldade. Por isso cada grafo tem as suas referências (acaso e popularidade), e a pergunta é
quanto cada modelo fica acima delas no próprio grafo. O script também conta as triplas do teste cuja
inversa (destino, tipo, origem) está no treino: um modelo que só aprende que a relação vale nos dois
sentidos já acerta essas.

No fim, mostra lado a lado o grafo das anotações (com os resultados já salvos dos experimentos 1, grade
ampliada, e 1c) e os grafos do 5g.

Saídas (não versionadas), para cada grafo:
  results/experimento6_<grafo>_resultados.json   o JSON do experimento 1, mais o grafo, as inversas e a
                                                 referência de popularidade
  results/experimento6_<grafo>_vetores/*.npz     vetores das entidades de cada modelo e semente
e a comparação do fim, com o grafo das anotações:
  results/experimento6_comparacao.json           tamanho, inversas, referências e resumo de cada grafo

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/06_embeddings_grafo_5g.py --rapido            # testa o código e estima o tempo
    python experimentos/06_embeddings_grafo_5g.py                     # os dois grafos, depois a comparação
    python experimentos/06_embeddings_grafo_5g.py --grafos filtrado   # um grafo só
    python experimentos/06_embeddings_grafo_5g.py --retomar           # continua uma rodada interrompida
    python experimentos/06_embeddings_grafo_5g.py --comparar          # só a comparação, com o que já foi salvo
"""
import argparse
import collections
import importlib.util
import json
import logging
import os
import sys
import warnings

import numpy as np
import pykeen
import torch
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_5G = os.path.join(RAIZ, "results", "experimento5g", "prontas")
GRAFOS = {"completo": "grafo_completo.tsv", "filtrado": "grafo_filtrado.tsv"}
GRADE = "ampliada"


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Divisão, treino e avaliação: os do experimento 1; referência de popularidade: a do 1c
retreino = carregar("retreino", "01_retreino_kge.py")
popularidade = carregar("popularidade", "01c_referencia_popularidade.py")
RESULTADOS_ANOTACOES = os.path.join(RAIZ, "results", f"{retreino.GRADES[GRADE]['prefixo']}_resultados.json")
SAIDA_COMPARACAO = os.path.join(RAIZ, "results", "experimento6_comparacao.json")


def prefixo(nome):
    return f"experimento6_{nome}"


def saida_json(nome):
    return os.path.join(RAIZ, "results", f"{prefixo(nome)}_resultados.json")


def ler_tsv(caminho):
    """Triplas (origem, tipo, destino) distintas e em ordem, como as do ler_triplas do experimento 1."""
    with open(caminho, encoding="utf-8") as f:
        triplas = {tuple(linha.rstrip("\r\n").split("\t")) for linha in f if linha.strip()}
    return np.array(sorted(triplas), dtype=str)


def dividir(triplas):
    fabrica = TriplesFactory.from_labeled_triples(triples=triplas)
    treino, validacao, teste = fabrica.split(list(retreino.DIVISAO), random_state=retreino.SEMENTE_DIVISAO)
    return fabrica, treino, validacao, teste


def inversas_no_treino(treino, teste):
    """Triplas do teste cuja inversa (destino, tipo, origem) está no treino."""
    vistas = set(map(tuple, treino.mapped_triples.tolist()))
    n = sum((t, r, h) in vistas for h, r, t in teste.mapped_triples.tolist())
    return {"triplas": n, "fracao": n / teste.num_triples}


def informacoes(nome, triplas, fabrica, treino, teste):
    """O que entra no JSON além do que o experimento 1 já guarda."""
    return {
        "grafo": {"nome": nome, "arquivo": os.path.relpath(os.path.join(PASTA_5G, GRAFOS[nome]), RAIZ),
                  "triplas": fabrica.num_triples, "entidades": fabrica.num_entities,
                  "tipos": dict(collections.Counter(triplas[:, 1].tolist()).most_common())},
        "inversas_no_treino": inversas_no_treino(treino, teste),
    }


def rodar(nome, rapido, retomar):
    triplas = ler_tsv(os.path.join(PASTA_5G, GRAFOS[nome]))
    fabrica, treino, validacao, teste = dividir(triplas)
    info = informacoes(nome, triplas, fabrica, treino, teste)
    inversas = info["inversas_no_treino"]
    print(f"\n######## Grafo {nome} do 5g: {fabrica.num_triples} triplas ({treino.num_triples} treino,"
          f" {validacao.num_triples} validação, {teste.num_triples} teste) | entidades: {fabrica.num_entities}")
    print("   tipos: " + ", ".join(f"{tipo} {n}" for tipo, n in info["grafo"]["tipos"].items()))
    print(f"   triplas do teste com a inversa no treino: {inversas['triplas']} ({inversas['fracao']:.1%})")
    if rapido:
        retreino.modo_rapido(treino, validacao, teste, GRADE)
        return
    referencias = popularidade.avaliar_referencias(treino, validacao, teste)
    print(popularidade.linha("acaso", retreino.referencia_ao_acaso(treino.num_entities)))
    for nome_referencia, r in referencias.items():
        print(popularidade.linha(nome_referencia, r["teste"]))
    retreino.modo_completo(GRADE, treino, validacao, teste, prefixo=prefixo(nome),
                           extras={**info, "referencias": referencias}, retomar=retomar)


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def colunas_da_comparacao():
    """Para cada grafo com resultados salvos: tamanho, inversas, referências e resumo dos modelos."""
    colunas = {}
    if os.path.isfile(RESULTADOS_ANOTACOES) and os.path.isfile(popularidade.SAIDA_JSON):
        treinados, referencias = ler_json(RESULTADOS_ANOTACOES), ler_json(popularidade.SAIDA_JSON)
        fabrica, treino, _, teste = dividir(retreino.ler_triplas(retreino.PASTA_XML))
        colunas["anotações"] = {
            "triplas": fabrica.num_triples, "entidades": fabrica.num_entities, "tipos": fabrica.num_relations,
            "inversas": inversas_no_treino(treino, teste), "acaso": treinados["referencia_ao_acaso"],
            "popularidade": referencias["resultados"]["popularidade"], "resumo": treinados["resumo"]}
    for nome in GRAFOS:
        if os.path.isfile(saida_json(nome)):
            d = ler_json(saida_json(nome))
            colunas[f"5g {nome}"] = {
                "triplas": d["grafo"]["triplas"], "entidades": d["grafo"]["entidades"],
                "tipos": len(d["grafo"]["tipos"]), "inversas": d["inversas_no_treino"],
                "acaso": d["referencia_ao_acaso"], "popularidade": d["referencias"]["popularidade"],
                "resumo": d["resumo"]}
    return colunas


def media(resumo, modelo, metrica):
    valor = resumo.get(modelo, {}).get("teste", {}).get(metrica)
    return None if valor is None else valor["media"]


def celula(valor, base=None):
    """Valor com 4 casas e, se houver base, quantas vezes ele é a base (a popularidade)."""
    if valor is None:
        return "—"
    return f"{valor:.4f}" + (f" ({valor / base:.2f}x)" if base else "")


def comparar():
    colunas = colunas_da_comparacao()
    if not colunas:
        print("\nNenhum resultado salvo para comparar.")
        return
    retreino.salvar_json(colunas, SAIDA_COMPARACAO)
    modelos = list(dict.fromkeys(m for c in colunas.values() for m in c["resumo"]))

    def linha(rotulo, valores):
        print(f"   {rotulo:20s}" + "".join(f"{v:>18s}" for v in valores))

    print("\n== Comparação entre os grafos: teste filtrado; modelos com a média das sementes e, entre parênteses,"
          " quantas vezes a popularidade")
    linha("", list(colunas))
    linha("triplas", [str(c["triplas"]) for c in colunas.values()])
    linha("entidades", [str(c["entidades"]) for c in colunas.values()])
    linha("tipos de relação", [str(c["tipos"]) for c in colunas.values()])
    linha("inversa no treino", [f"{c['inversas']['fracao']:.1%}" for c in colunas.values()])
    for metrica in ("MRR", "Hits@10"):
        print(f"   {metrica}")
        linha("  acaso", [celula(c["acaso"][metrica]) for c in colunas.values()])
        linha("  popularidade", [celula(c["popularidade"]["teste"][metrica]) for c in colunas.values()])
        for modelo in modelos:
            linha(f"  {modelo}", [celula(media(c["resumo"], modelo, metrica), c["popularidade"]["teste"][metrica])
                                  for c in colunas.values()])

    for nome, c in colunas.items():
        modelos_aqui = [m for m in modelos if m in c["resumo"]]
        print(f"\n== MRR por tipo de relação, grafo {nome} (modelos: média das sementes)")
        print(f"   {'tipo':16s} {'triplas':>8s} {'popular.':>9s}" + "".join(f" {m:>10s}" for m in modelos_aqui))
        por_tipo = c["popularidade"]["teste_por_relacao"]
        for tipo, p in sorted(por_tipo.items(), key=lambda item: -item[1]["triplas"]):
            valores = []
            for m in modelos_aqui:
                valor = c["resumo"][m]["teste_por_relacao"].get(tipo, {}).get("MRR")
                valores.append("—" if valor is None else f"{valor['media']:.4f}")
            print(f"   {tipo:16s} {p['triplas']:8d} {p['MRR']:9.4f}" + "".join(f" {v:>10s}" for v in valores))
    print(f"\nComparação em {os.path.relpath(SAIDA_COMPARACAO, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--rapido", action="store_true", help="poucas épocas: testa o código e estima o tempo")
    argumentos.add_argument("--grafos", nargs="+", choices=list(GRAFOS), default=list(GRAFOS),
                            help="grafos do 5g (padrão: os dois)")
    argumentos.add_argument("--retomar", action="store_true",
                            help="não refaz as execuções já salvas no JSON de cada grafo (depois de uma interrupção)")
    argumentos.add_argument("--comparar", action="store_true", help="só a comparação, com os resultados já salvos")
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    args = argumentos.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)
    print(f"PyKEEN {pykeen.get_version()} | PyTorch {torch.__version__} | {args.threads} núcleos | grade {GRADE}")

    if not args.comparar:
        for nome in args.grafos:
            rodar(nome, args.rapido, args.retomar)
    if not args.rapido:
        comparar()


if __name__ == "__main__":
    main()
