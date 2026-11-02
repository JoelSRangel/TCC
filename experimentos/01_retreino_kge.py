"""
Experimento 1: retreino de TransE, ComplEx e RotatE com um regime adequado.

O benchmark anterior treinou os três modelos por 10 épocas, com perda por
margem, 1 exemplo negativo por positivo e sem ajuste de hiperparâmetros
(ver analises/07_configuracao_pykeen.py). Só o RotatE ficou bem acima do
acaso. Este experimento testa a hipótese de que o regime de treino explica
o resultado.

Etapas:
  1. Divisão das triplas em treino, validação e teste (80/10/10, semente 42).
  2. Seleção: para cada modelo, testa todas as configurações da grade, com
     a semente 0, até 300 épocas e parada antecipada pela validação: o MRR
     da validação é medido a cada 10 épocas, e o treino para depois de 3
     medições seguidas sem melhora de pelo menos 1%. Fica a configuração
     com o maior MRR na validação.
  3. Avaliação final: a configuração escolhida de cada modelo é treinada
     com as sementes 0 a 4 e avaliada no teste, filtrado pelas triplas de
     treino e de validação, no total e separado por tipo de relação. Por
     relação, também separa a previsão da origem e a do destino.

Duas grades de configurações:
  basica   perda por margem ou softplus; 1 ou 16 negativos; taxa de
           aprendizado 0,001. Modelos: TransE, ComplEx e RotatE.
  ampliada perda por margem, softplus ou adversarial (NSSA, a do artigo do
           RotatE); 1, 16 ou 64 negativos; taxa 0,001 ou 0,01. Modelos:
           os três, mais o TransE com 128 dimensões, que tem o mesmo
           número de valores por entidade do ComplEx e do RotatE.

O treino sempre termina com os pesos da melhor época na validação: o
PyKEEN os recarrega quando a parada antecipada interrompe o treino, e este
script os recarrega quando o treino chega ao máximo de épocas sem parar.

As triplas vêm dos XMLs, com a mesma chave do grafo do Neo4j (texto
normalizado + categoria), então não é preciso o Neo4j rodando.

Saídas (não versionadas), conforme a grade:
  results/experimento1_resultados.json            (basica)
  results/experimento1_ampliado_resultados.json   (ampliada)
  results/<mesmo prefixo>_vetores/*.npz           vetores das entidades de
                                                  cada modelo e semente

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/01_retreino_kge.py --rapido            # testa o código e estima o tempo
    python experimentos/01_retreino_kge.py                     # grade básica
    python experimentos/01_retreino_kge.py --grade ampliada    # grade ampliada
"""
import argparse
import glob
import itertools
import json
import logging
import os
import platform
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

GRADES = {
    "basica": {
        "modelos": [("TransE", "TransE", 64), ("ComplEx", "ComplEx", 64), ("RotatE", "RotatE", 64)],
        "perdas": ("marginranking", "softplus"),
        "negativos": (1, 16),
        "taxas": (0.001,),
        "prefixo": "experimento1",
    },
    "ampliada": {
        "modelos": [("TransE", "TransE", 64), ("TransE-128", "TransE", 128),
                    ("ComplEx", "ComplEx", 64), ("RotatE", "RotatE", 64)],
        "perdas": ("marginranking", "softplus", "nssa"),
        "negativos": (1, 16, 64),
        "taxas": (0.001, 0.01),
        "prefixo": "experimento1_ampliado",
    },
}
SEMENTE_SELECAO = 0
SEMENTES_FINAIS = (0, 1, 2, 3, 4)
DIVISAO, SEMENTE_DIVISAO = (0.8, 0.1, 0.1), 42
LOTE, MAX_EPOCAS = 256, 300
PARADA = dict(frequency=10, patience=3, relative_delta=0.01, metric="inverse_harmonic_mean_rank")
METRICAS = {"MRR": "inverse_harmonic_mean_rank", "Hits@1": "hits_at_1", "Hits@3": "hits_at_3",
            "Hits@10": "hits_at_10", "AMRI": "adjusted_arithmetic_mean_rank_index"}
# Além da média dos dois lados ("both"), guarda MRR e Hits@10 ao prever a
# origem ("head") e ao prever o destino ("tail").
LADOS = {"head": "origem", "tail": "destino"}
METRICAS_LADO = ("MRR", "Hits@10")
CHAVES_RESUMO = list(METRICAS) + [f"{m}_{nome}" for nome in LADOS.values() for m in METRICAS_LADO]


def chave(texto, tag):
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def ler_triplas(pasta):
    """Triplas (origem, tipo, destino) distintas, como no grafo do Neo4j."""
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
    """Métricas na variante 'realistic', casando a chave exata do PyKEEN."""
    plano = resultado.to_flat_dict()
    saida = {}
    for nome, metrica in METRICAS.items():
        valor = plano.get(f"both.realistic.{metrica}")
        saida[nome] = None if valor is None else float(valor)
    for lado, nome_lado in LADOS.items():
        for nome in METRICAS_LADO:
            valor = plano.get(f"{lado}.realistic.{METRICAS[nome]}")
            saida[f"{nome}_{nome_lado}"] = None if valor is None else float(valor)
    if saida["MRR"] is None:
        disponiveis = sorted(k for k in plano if k.startswith("both.realistic."))
        raise KeyError(f"métrica não encontrada; chaves disponíveis: {disponiveis}")
    return saida


def treinar(modelo, dimensao, perda, negativos, taxa, semente, treino, validacao, max_epocas):
    """Treina com parada antecipada e devolve o modelo com os pesos da melhor época."""
    inicio = time.time()
    resultado = pipeline(
        training=treino,
        validation=validacao,
        testing=validacao,  # a avaliação de teste é feita depois, com os pesos da melhor época
        model=modelo,
        model_kwargs=dict(embedding_dim=dimensao),
        loss=perda,
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=negativos),
        optimizer_kwargs=dict(lr=taxa),
        training_kwargs=dict(num_epochs=max_epocas, batch_size=LOTE, use_tqdm=False),
        stopper="early",
        stopper_kwargs=dict(PARADA),
        evaluation_kwargs=dict(batch_size=LOTE, use_tqdm=False),
        random_seed=semente,
        device="cpu",
    )
    parada = resultado.stopper
    caminho = getattr(parada, "best_model_path", None)
    if caminho is not None and caminho.is_file():
        # Chegou ao máximo de épocas sem parar: o PyKEEN não recarrega os pesos sozinho.
        resultado.model.load_state_dict(torch.load(caminho, weights_only=False))
        caminho.unlink()
    info = {
        "epocas_treinadas": len(resultado.losses),
        "melhor_epoca": parada.best_epoch,
        "mrr_validacao": None if parada.best_metric is None else float(parada.best_metric),
        "segundos": round(time.time() - inicio, 1),
    }
    return resultado.model, info


def avaliar(modelo, triplas, filtros):
    avaliador = RankBasedEvaluator()  # filtrado
    resultado = avaliador.evaluate(model=modelo, mapped_triples=triplas, additional_filter_triples=filtros,
                                   batch_size=LOTE, use_tqdm=False)
    return metricas(resultado)


def avaliar_teste(modelo, treino, validacao, teste):
    filtros = [treino.mapped_triples, validacao.mapped_triples]
    total = avaliar(modelo, teste.mapped_triples, filtros)
    por_relacao = {}
    for relacao, indice in sorted(treino.relation_to_id.items()):
        subconjunto = teste.mapped_triples[teste.mapped_triples[:, 1] == indice]
        por_relacao[relacao] = {"triplas": int(len(subconjunto)), **avaliar(modelo, subconjunto, filtros)}
    return total, por_relacao


def salvar_vetores(modelo, treino, pasta, rotulo, semente):
    os.makedirs(pasta, exist_ok=True)
    vetores = modelo.entity_representations[0](indices=None).detach().cpu().numpy()
    rotulos = [None] * treino.num_entities
    for entidade, indice in treino.entity_to_id.items():
        rotulos[indice] = entidade
    caminho = os.path.join(pasta, f"{rotulo}_semente{semente}.npz")
    np.savez_compressed(caminho, ids=np.array(rotulos), vetores=vetores)
    return os.path.relpath(caminho, RAIZ)


def resumir(valores):
    validos = [v for v in valores if v is not None]
    if not validos:
        return None
    desvio = statistics.pstdev(validos) if len(validos) > 1 else 0.0
    return {"media": statistics.fmean(validos), "desvio": desvio, "n": len(validos)}


def referencia_ao_acaso(n):
    return {"MRR": sum(1 / k for k in range(1, n + 1)) / n,
            "Hits@1": 1 / n, "Hits@3": 3 / n, "Hits@10": 10 / n, "AMRI": 0.0}


def salvar_json(dados, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def chave_execucao(registro):
    """Modelo, configuração e semente de uma execução (na seleção, a semente fica None)."""
    return tuple(registro.get(k) for k in ("modelo", "dimensao", "perda", "negativos", "taxa", "semente"))


def execucoes_salvas(dados, caminho):
    """As execuções sem erro já gravadas no JSON de uma rodada interrompida, se o protocolo for o mesmo."""
    if not os.path.isfile(caminho):
        return {}
    with open(caminho, encoding="utf-8") as f:
        anterior = json.load(f)
    for campo in ("grade", "versoes", "protocolo", "triplas"):
        if anterior.get(campo) != json.loads(json.dumps(dados[campo])):
            raise ValueError(f"o campo {campo} de {os.path.relpath(caminho, RAIZ)} é outro; não dá para retomar")
    return {chave_execucao(r): r for r in anterior["selecao"] + anterior["final"] if "erro" not in r}


def descricao(c):
    return f"{c['perda']:13s} {c['negativos']:2d} neg. taxa {c['taxa']}"


def modo_rapido(treino, validacao, teste, nome_grade="basica"):
    """Treina cada modelo da grade por poucas épocas, testa todo o código e estima o tempo da grade."""
    grade = GRADES[nome_grade]
    epocas = 20
    print(f"\nModo rápido: 1 configuração por modelo (softplus, 16 negativos), {epocas} épocas, semente 0.")
    print("Os números abaixo servem só para conferir o código; não são resultados.\n")
    estimativas = {}
    for rotulo, modelo_pykeen, dimensao in grade["modelos"]:
        modelo, info = treinar(modelo_pykeen, dimensao, "softplus", 16, 0.001, 0, treino, validacao, epocas)
        total, por_relacao = avaliar_teste(modelo, treino, validacao, teste)
        por_epoca = info["segundos"] / max(info["epocas_treinadas"], 1)
        estimativas[rotulo] = por_epoca
        print(f"   {rotulo:8s} {info['epocas_treinadas']} épocas em {info['segundos']} s"
              f" ({por_epoca:.2f} s por época, com as avaliações de validação)")
        print(f"            teste: MRR {total['MRR']:.4f} | Hits@10 {total['Hits@10']:.4f} | AMRI {total['AMRI']:.4f}")
        for relacao, m in por_relacao.items():
            print(f"            {relacao}: {m['triplas']} triplas, MRR {m['MRR']:.4f}"
                  f" (origem {m['MRR_origem']:.4f}, destino {m['MRR_destino']:.4f})")
    execucoes = len(grade["perdas"]) * len(grade["negativos"]) * len(grade["taxas"]) + len(SEMENTES_FINAIS)
    pior = sum(execucoes * MAX_EPOCAS * t for t in estimativas.values())
    provavel = sum(execucoes * 100 * t for t in estimativas.values())
    print(f"\nEstimativa da grade {nome_grade} ({execucoes} execuções por modelo):")
    print(f"   se a parada antecipada agir por volta da época 100: {provavel / 3600:.1f} h")
    print(f"   pior caso, todas as execuções até {MAX_EPOCAS} épocas: {pior / 3600:.1f} h")


def modo_completo(nome_grade, treino, validacao, teste, prefixo=None, extras=None, retomar=False):
    """Seleção e avaliação final da grade. Com prefixo e extras, outro experimento (o 6) usa o mesmo
    protocolo em outro grafo, com as saídas em outros arquivos e informações a mais no JSON. Com
    retomar, as execuções já gravadas no JSON (de uma rodada interrompida) não são treinadas de novo."""
    grade = GRADES[nome_grade]
    prefixo = prefixo or grade["prefixo"]
    saida_json = os.path.join(RAIZ, "results", f"{prefixo}_resultados.json")
    pasta_vetores = os.path.join(RAIZ, "results", f"{prefixo}_vetores")
    configuracoes = [{"perda": p, "negativos": n, "taxa": t}
                     for p, n, t in itertools.product(grade["perdas"], grade["negativos"], grade["taxas"])]
    dados = {
        "grade": nome_grade,
        "versoes": {"python": platform.python_version(), "pykeen": pykeen.get_version(),
                    "torch": torch.__version__},
        "protocolo": {"divisao": DIVISAO, "semente_divisao": SEMENTE_DIVISAO, "lote": LOTE,
                      "max_epocas": MAX_EPOCAS, "parada_antecipada": PARADA,
                      "modelos": grade["modelos"], "configuracoes": configuracoes,
                      "semente_selecao": SEMENTE_SELECAO, "sementes_finais": SEMENTES_FINAIS},
        "triplas": {"treino": treino.num_triples, "validacao": validacao.num_triples,
                    "teste": teste.num_triples, "entidades": treino.num_entities},
        "referencia_ao_acaso": referencia_ao_acaso(treino.num_entities),
        **(extras or {}),
        "selecao": [], "final": [], "resumo": {},
    }
    salvas = execucoes_salvas(dados, saida_json) if retomar else {}
    print(f"Grade {nome_grade}: {len(configuracoes)} configurações por modelo,"
          f" {len(grade['modelos'])} modelos")
    if retomar:
        print(f"Retomada: {len(salvas)} execuções já salvas em {os.path.relpath(saida_json, RAIZ)}")
    for rotulo, modelo_pykeen, dimensao in grade["modelos"]:
        print(f"\n== {rotulo} (dimensão {dimensao}): seleção da configuração (semente {SEMENTE_SELECAO})")
        candidatos = []
        for c in configuracoes:
            registro = {"modelo": rotulo, "dimensao": dimensao, **c}
            salva = salvas.get(chave_execucao(registro))
            if salva is not None:  # treinada antes da interrupção
                candidatos.append(salva)
                dados["selecao"].append(salva)
                print(f"   {descricao(c)}: MRR validação {salva['mrr_validacao']:.4f} (já salva)")
                continue
            try:
                _, info = treinar(modelo_pykeen, dimensao, c["perda"], c["negativos"], c["taxa"],
                                  SEMENTE_SELECAO, treino, validacao, MAX_EPOCAS)
                registro.update(info)
                if info["mrr_validacao"] is None:
                    raise ValueError("a parada antecipada não registrou nenhum MRR de validação")
                candidatos.append(registro)
                print(f"   {descricao(c)}: MRR validação {info['mrr_validacao']:.4f}"
                      f" (melhor época {info['melhor_epoca']}, {info['epocas_treinadas']} épocas, {info['segundos']} s)")
            except Exception as erro:  # registra a falha em vez de contar como zero
                registro["erro"] = repr(erro)
                print(f"   {descricao(c)}: FALHOU ({erro})")
            dados["selecao"].append(registro)
            salvar_json(dados, saida_json)
        if not candidatos:
            print(f"   nenhuma configuração do {rotulo} terminou; modelo fora do resumo")
            continue
        escolhida = max(candidatos, key=lambda r: r["mrr_validacao"])
        config = {k: escolhida[k] for k in ("perda", "negativos", "taxa")}
        print(f"   escolhida: {descricao(config)}")

        print(f"== {rotulo}: avaliação final no teste (sementes {list(SEMENTES_FINAIS)})")
        execucoes = []
        for semente in SEMENTES_FINAIS:
            registro = {"modelo": rotulo, "dimensao": dimensao, **config, "semente": semente}
            salva = salvas.get(chave_execucao(registro))
            if salva is not None:
                execucoes.append(salva)
                dados["final"].append(salva)
                print(f"   semente {semente}: MRR {salva['teste']['MRR']:.4f} (já salva)")
                continue
            try:
                modelo, info = treinar(modelo_pykeen, dimensao, config["perda"], config["negativos"],
                                       config["taxa"], semente, treino, validacao, MAX_EPOCAS)
                total, por_relacao = avaliar_teste(modelo, treino, validacao, teste)
                registro.update(info, teste=total, teste_por_relacao=por_relacao,
                                vetores=salvar_vetores(modelo, treino, pasta_vetores, rotulo, semente))
                execucoes.append(registro)
                print(f"   semente {semente}: MRR {total['MRR']:.4f} | Hits@10 {total['Hits@10']:.4f}"
                      f" | AMRI {total['AMRI']:.4f} (melhor época {info['melhor_epoca']}, {info['segundos']} s)")
            except Exception as erro:
                registro["erro"] = repr(erro)
                print(f"   semente {semente}: FALHOU ({erro})")
            dados["final"].append(registro)
            salvar_json(dados, saida_json)

        relacoes = sorted(execucoes[0]["teste_por_relacao"]) if execucoes else []
        dados["resumo"][rotulo] = {
            "dimensao": dimensao,
            "configuracao": config,
            "execucoes_validas": len(execucoes),
            "teste": {m: resumir([e["teste"][m] for e in execucoes]) for m in CHAVES_RESUMO},
            "teste_por_relacao": {r: {m: resumir([e["teste_por_relacao"][r][m] for e in execucoes])
                                      for m in CHAVES_RESUMO} for r in relacoes},
        }
        salvar_json(dados, saida_json)

    print("\n== Resumo no teste (média ± desvio padrão entre as sementes)")
    acaso = dados["referencia_ao_acaso"]
    print(f"   {'acaso':10s} MRR {acaso['MRR']:.4f} | Hits@10 {acaso['Hits@10']:.4f} | AMRI 0")
    for rotulo, r in dados["resumo"].items():
        t = r["teste"]
        if t["MRR"] is None:
            continue
        print(f"   {rotulo:10s} MRR {t['MRR']['media']:.4f} ± {t['MRR']['desvio']:.4f}"
              f" | Hits@10 {t['Hits@10']['media']:.4f} ± {t['Hits@10']['desvio']:.4f}"
              f" | AMRI {t['AMRI']['media']:.4f} ({r['execucoes_validas']} execuções válidas;"
              f" {descricao(r['configuracao'])})")
    print(f"\nResultados completos em {os.path.relpath(saida_json, RAIZ)}")
    return dados


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--rapido", action="store_true", help="poucas épocas: testa o código e estima o tempo")
    argumentos.add_argument("--grade", choices=sorted(GRADES), default="basica", help="grade de configurações")
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    args = argumentos.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)

    triplas = ler_triplas(PASTA_XML)
    fabrica = TriplesFactory.from_labeled_triples(triples=triplas)
    treino, validacao, teste = fabrica.split(list(DIVISAO), random_state=SEMENTE_DIVISAO)
    print(f"PyKEEN {pykeen.get_version()} | PyTorch {torch.__version__} | {args.threads} núcleos")
    print(f"Triplas: {fabrica.num_triples} ({treino.num_triples} treino, {validacao.num_triples} validação,"
          f" {teste.num_triples} teste) | entidades: {fabrica.num_entities}")

    if args.rapido:
        modo_rapido(treino, validacao, teste, args.grade)
    else:
        modo_completo(args.grade, treino, validacao, teste)


if __name__ == "__main__":
    main()
