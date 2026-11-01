"""
Experimento 1b: por que o ComplEx não aprende neste grafo?

No experimento 1, o ComplEx ficou no nível do acaso nas quatro
configurações testadas (perda por margem ou softplus; 1 ou 16 negativos),
mesmo com até 300 épocas e parada antecipada. Em todas elas ele usou a
regularização padrão do PyKEEN: L2 com peso 0,01 nas entidades e relações
(ver analises/07_configuracao_pykeen.py).

Este experimento testa o ComplEx sem essa regularização, e também com o
treino em que ele costuma ir bem na literatura: cada par (origem, relação)
é comparado com todas as entidades de uma vez (laço LCWA do PyKEEN), com
perda logística (BCE). Configurações testadas, todas sem regularização:
   1-4. laço sLCWA, perda softplus ou margem, 1 ou 16 negativos, taxa 0,001
   5.   laço sLCWA, softplus, 16 negativos, taxa de aprendizado 0,01
   6-7. laço LCWA, perda BCE, taxa de aprendizado 0,001 ou 0,01

A divisão das triplas, a dimensão, o lote, o limite de épocas e a parada
antecipada são os mesmos do experimento 1, então os resultados são
comparáveis. A configuração com o maior MRR na validação (semente 0) é
treinada com as sementes 0 a 4 e avaliada no teste, no total e por tipo
de relação.

Saídas (não versionadas):
  results/experimento1b_complex.json
  results/experimento1b_vetores/ComplEx_semente*.npz

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/01b_complex.py
"""
import argparse
import glob
import json
import logging
import os
import statistics
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pykeen
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = os.path.join(RAIZ, "data", "raw")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento1b_complex.json")
PASTA_VETORES = os.path.join(RAIZ, "results", "experimento1b_vetores")

# Iguais ao experimento 1
DIVISAO, SEMENTE_DIVISAO = (0.8, 0.1, 0.1), 42
DIMENSAO, LOTE, MAX_EPOCAS = 64, 256, 300
PARADA = dict(frequency=10, patience=3, relative_delta=0.01, metric="inverse_harmonic_mean_rank")
SEMENTE_SELECAO, SEMENTES_FINAIS = 0, (0, 1, 2, 3, 4)
METRICAS = {"MRR": "inverse_harmonic_mean_rank", "Hits@1": "hits_at_1", "Hits@3": "hits_at_3",
            "Hits@10": "hits_at_10", "AMRI": "adjusted_arithmetic_mean_rank_index"}

# Configurações testadas (todas sem regularização)
CONFIGURACOES = [
    {"laco": "slcwa", "perda": "softplus", "negativos": 1, "taxa": 0.001},
    {"laco": "slcwa", "perda": "softplus", "negativos": 16, "taxa": 0.001},
    {"laco": "slcwa", "perda": "marginranking", "negativos": 1, "taxa": 0.001},
    {"laco": "slcwa", "perda": "marginranking", "negativos": 16, "taxa": 0.001},
    {"laco": "slcwa", "perda": "softplus", "negativos": 16, "taxa": 0.01},
    {"laco": "lcwa", "perda": "bcewithlogits", "negativos": None, "taxa": 0.001},
    {"laco": "lcwa", "perda": "bcewithlogits", "negativos": None, "taxa": 0.01},
]


def chave(texto, tag):
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def ler_triplas(pasta):
    triplas = set()
    for arquivo in sorted(glob.glob(os.path.join(pasta, "*.xml"))):
        raiz = ET.parse(arquivo).getroot()
        local = {a.get("id"): chave(a.get("text"), a.get("tag")) for a in raiz.findall(".//annotation")}
        for rel in raiz.findall(".//rel"):
            origem = local.get(rel.get("annotation1"))
            destino = local.get(rel.get("annotation2"))
            if origem and destino:
                triplas.add((origem, rel.get("reltype"), destino))
    return np.array(sorted(triplas), dtype=str)


def metricas(resultado):
    plano = resultado.to_flat_dict()
    return {nome: (None if plano.get(f"both.realistic.{m}") is None else float(plano[f"both.realistic.{m}"]))
            for nome, m in METRICAS.items()}


def descricao(c):
    if c["laco"] == "lcwa":
        return f"LCWA, BCE, taxa {c['taxa']}"
    return f"sLCWA, {c['perda']}, {c['negativos']} neg., taxa {c['taxa']}"


def treinar(c, semente, treino, validacao):
    inicio = time.time()
    extra = dict(loss=c["perda"], training_loop=c["laco"])
    if c["laco"] == "slcwa":
        extra.update(negative_sampler="basic", negative_sampler_kwargs=dict(num_negs_per_pos=c["negativos"]))
    resultado = pipeline(
        training=treino, validation=validacao, testing=validacao,
        model="ComplEx", model_kwargs=dict(embedding_dim=DIMENSAO, regularizer=None),
        optimizer_kwargs=dict(lr=c["taxa"]),
        training_kwargs=dict(num_epochs=MAX_EPOCAS, batch_size=LOTE, use_tqdm=False),
        stopper="early", stopper_kwargs=dict(PARADA),
        evaluation_kwargs=dict(batch_size=LOTE, use_tqdm=False),
        random_seed=semente, device="cpu", **extra,
    )
    parada = resultado.stopper
    caminho = getattr(parada, "best_model_path", None)
    if caminho is not None and caminho.is_file():
        # Chegou ao máximo de épocas sem parar: recarrega os pesos da melhor época.
        resultado.model.load_state_dict(torch.load(caminho, weights_only=False))
        caminho.unlink()
    regularizacao = [type(r.regularizer).__name__ if r.regularizer is not None else None
                     for r in resultado.model.entity_representations]
    info = {"epocas_treinadas": len(resultado.losses), "melhor_epoca": parada.best_epoch,
            "mrr_validacao": None if parada.best_metric is None else float(parada.best_metric),
            "regularizacao": regularizacao, "segundos": round(time.time() - inicio, 1)}
    return resultado.model, info


def avaliar(modelo, triplas, filtros):
    r = RankBasedEvaluator().evaluate(model=modelo, mapped_triples=triplas, additional_filter_triples=filtros,
                                      batch_size=LOTE, use_tqdm=False)
    return metricas(r)


def resumir(valores):
    v = [x for x in valores if x is not None]
    if not v:
        return None
    return {"media": statistics.fmean(v), "desvio": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def salvar(dados):
    os.makedirs(os.path.dirname(SAIDA_JSON), exist_ok=True)
    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)

    fabrica = TriplesFactory.from_labeled_triples(triples=ler_triplas(PASTA_XML))
    treino, validacao, teste = fabrica.split(list(DIVISAO), random_state=SEMENTE_DIVISAO)
    n = treino.num_entities
    acaso = sum(1 / k for k in range(1, n + 1)) / n
    print(f"PyKEEN {pykeen.get_version()} | {args.threads} núcleos | {treino.num_triples} treino,"
          f" {validacao.num_triples} validação, {teste.num_triples} teste | MRR ao acaso {acaso:.4f}")

    dados = {"configuracoes": CONFIGURACOES, "mrr_ao_acaso": acaso, "selecao": [], "final": [], "resumo": None}
    print(f"\n== Seleção, sem regularização (semente {SEMENTE_SELECAO})")
    candidatos = []
    for c in CONFIGURACOES:
        registro = dict(c)
        try:
            _, info = treinar(c, SEMENTE_SELECAO, treino, validacao)
            registro.update(info)
            candidatos.append(registro)
            print(f"   {descricao(c):38s} MRR validação {info['mrr_validacao']:.4f}"
                  f" ({info['mrr_validacao'] / acaso:.1f} vezes o acaso; melhor época {info['melhor_epoca']},"
                  f" {info['segundos']} s; regularização: {info['regularizacao']})")
        except Exception as erro:
            registro["erro"] = repr(erro)
            print(f"   {descricao(c):38s} FALHOU ({erro})")
        dados["selecao"].append(registro)
        salvar(dados)
    if not candidatos:
        print("Nenhuma configuração terminou.")
        return
    escolhida = max(candidatos, key=lambda r: r["mrr_validacao"])
    c = {k: escolhida[k] for k in ("laco", "perda", "negativos", "taxa")}
    print(f"   escolhida: {descricao(c)}")

    print(f"\n== Avaliação final no teste (sementes {list(SEMENTES_FINAIS)})")
    filtros = [treino.mapped_triples, validacao.mapped_triples]
    execucoes = []
    os.makedirs(PASTA_VETORES, exist_ok=True)
    rotulos = [None] * n
    for rotulo, indice in treino.entity_to_id.items():
        rotulos[indice] = rotulo
    for semente in SEMENTES_FINAIS:
        registro = dict(c, semente=semente)
        try:
            modelo, info = treinar(c, semente, treino, validacao)
            total = avaliar(modelo, teste.mapped_triples, filtros)
            por_relacao = {}
            for relacao, indice in sorted(treino.relation_to_id.items()):
                sub = teste.mapped_triples[teste.mapped_triples[:, 1] == indice]
                por_relacao[relacao] = {"triplas": int(len(sub)), **avaliar(modelo, sub, filtros)}
            caminho = os.path.join(PASTA_VETORES, f"ComplEx_semente{semente}.npz")
            vetores = modelo.entity_representations[0](indices=None).detach().cpu().numpy()
            np.savez_compressed(caminho, ids=np.array(rotulos), vetores=vetores)
            registro.update(info, teste=total, teste_por_relacao=por_relacao, vetores=os.path.relpath(caminho, RAIZ))
            execucoes.append(registro)
            print(f"   semente {semente}: MRR {total['MRR']:.4f} | Hits@10 {total['Hits@10']:.4f}"
                  f" | AMRI {total['AMRI']:.4f} (melhor época {info['melhor_epoca']}, {info['segundos']} s)")
        except Exception as erro:
            registro["erro"] = repr(erro)
            print(f"   semente {semente}: FALHOU ({erro})")
        dados["final"].append(registro)
        salvar(dados)

    if execucoes:
        relacoes = sorted(execucoes[0]["teste_por_relacao"])
        dados["resumo"] = {
            "configuracao": c, "execucoes_validas": len(execucoes),
            "teste": {m: resumir([e["teste"][m] for e in execucoes]) for m in METRICAS},
            "teste_por_relacao": {r: {m: resumir([e["teste_por_relacao"][r][m] for e in execucoes])
                                      for m in METRICAS} for r in relacoes},
        }
        salvar(dados)
        t = dados["resumo"]["teste"]
        print(f"\n== Resumo no teste: MRR {t['MRR']['media']:.4f} ± {t['MRR']['desvio']:.4f}"
              f" ({t['MRR']['media'] / acaso:.1f} vezes o acaso) | Hits@10 {t['Hits@10']['media']:.4f}"
              f" | AMRI {t['AMRI']['media']:.4f} ± {t['AMRI']['desvio']:.4f}")
    print(f"Resultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


if __name__ == "__main__":
    main()
