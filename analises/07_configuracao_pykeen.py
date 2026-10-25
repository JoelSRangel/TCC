"""
Configuração que o PyKEEN de fato usou para treinar o TransE, o ComplEx e o RotatE.

Os benchmarks (compare_models.py e cluster_benchmark.py) chamam pipeline()
informando só a dimensão (64), as épocas (10), o tamanho do lote (256), a
semente e o dispositivo (CPU). Todo o resto (função de perda, otimizador,
taxa de aprendizado, amostragem negativa, regularização, avaliação) vem dos
valores padrão do PyKEEN, que podem mudar entre versões.

Este script não confia na documentação: ele roda o mesmo pipeline() com os
mesmos argumentos e inspeciona os objetos que o PyKEEN criou. Para ser
rápido, treina 1 época em vez de 10; isso não muda nenhuma configuração.

As triplas são montadas a partir dos XMLs com a mesma chave do Neo4j
(texto normalizado + categoria), então não é preciso o Neo4j rodando. A
divisão treino/teste pode não ser idêntica à do benchmark, porque a ordem
das triplas vem dos XMLs e não da consulta ao Neo4j; isso também não muda
a configuração.

Precisa do PyKEEN, então roda no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python analises/07_configuracao_pykeen.py
Leva cerca de 1 a 2 minutos.
"""
import glob
import inspect
import logging
import os
import platform
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pykeen
import torch
from pykeen.evaluation import evaluator_resolver
from pykeen.pipeline import pipeline
from pykeen.sampling import negative_sampler_resolver
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, "data", "raw")
MODELOS = ("TransE", "ComplEx", "RotatE")


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


def nome(obj):
    if obj is None:
        return "nenhum"
    return getattr(obj, "__name__", None) or type(obj).__name__


def atributos_simples(obj):
    """Atributos públicos de tipos simples (números, texto, booleanos)."""
    if obj is None:
        return ""
    itens = []
    for k, v in vars(obj).items():
        if not k.startswith("_") and k != "training" and isinstance(v, (int, float, str, bool)):
            itens.append(f"{k}={v}")
    # Alguns valores (como o peso da regularização) ficam guardados como
    # tensores do PyTorch, fora de vars(); estes são lidos à parte.
    for k in ("weight", "margin"):
        v = getattr(obj, k, None)
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            itens.append(f"{k}={float(v):g}")
    return ", ".join(itens)


def descrever_representacao(rotulo, rep):
    partes = [f"forma={tuple(rep.shape)}"]
    tipo = getattr(rep, "dtype", None)
    if tipo is not None:
        partes.append(f"tipo={tipo}")
    print(f"      {rotulo}: {', '.join(partes)}")
    print(f"         restrição (constrainer): {nome(getattr(rep, 'constrainer', None))}")
    reg = getattr(rep, "regularizer", None)
    print(f"         regularização: {nome(reg)} {atributos_simples(reg)}".rstrip())


def descrever(resultado, treino):
    modelo = resultado.model
    # No PyKEEN 1.11, o otimizador e o amostrador negativo não ficam guardados
    # no laço de treinamento depois do treino; a configuração efetiva fica em
    # resultado.configuration.
    config = resultado.configuration
    print(f"   função de perda: {nome(modelo.loss)} ({atributos_simples(modelo.loss)})")
    interacao = getattr(modelo, "interaction", None)
    print(f"   função de pontuação: {nome(interacao)} ({atributos_simples(interacao)})")
    print("   representações:")
    for i, rep in enumerate(modelo.entity_representations):
        descrever_representacao(f"entidades[{i}]", rep)
    for i, rep in enumerate(modelo.relation_representations):
        descrever_representacao(f"relações[{i}]", rep)
    reg_modelo = getattr(modelo, "regularizer", None)
    if reg_modelo is not None:
        print(f"   regularização do modelo: {nome(reg_modelo)} ({atributos_simples(reg_modelo)})")
    print(f"   laço de treinamento: {config.get('training_loop')}"
          f" (argumentos extras: {config.get('training_loop_kwargs') or 'nenhum'})")
    # O laço só cria o amostrador negativo durante o treino. Como os
    # benchmarks não informaram nenhum, ele usa o padrão do resolvedor;
    # aqui o padrão é criado do mesmo jeito, sobre as triplas de treino.
    amostrador = negative_sampler_resolver.make(
        None, mapped_triples=treino.mapped_triples,
        num_entities=treino.num_entities, num_relations=treino.num_relations)
    print(f"   amostragem negativa (padrão): {nome(amostrador)}"
          f" (negativos por positivo: {amostrador.num_negs_per_pos})")
    kw = config.get("optimizer_kwargs") or {}
    print(f"   otimizador: {config.get('optimizer')} (taxa de aprendizado={kw.get('lr')},"
          f" betas={kw.get('betas')}, weight_decay={kw.get('weight_decay')})")
    avaliacao = config.get("evaluation_kwargs") or {}
    filtros = list((avaliacao.get("additional_filter_triples") or {}).keys())
    print(f"   avaliação: {config.get('evaluator')}, filtrada removendo as triplas de: {filtros or 'nenhuma'}")
    print(f"   épocas: {config.get('num_epochs')} | lote: {config.get('batch_size')}"
          f" | parada antecipada: {nome(resultado.stopper)}")
    print(f"   semente: {resultado.random_seed} | dispositivo: {modelo.device}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    print("Versões")
    print(f"   Python {platform.python_version()} | PyKEEN {pykeen.get_version()}"
          f" | PyTorch {torch.__version__} | NumPy {np.__version__}")

    padroes = inspect.signature(pipeline).parameters
    print("\nPadrões do pipeline() para o que os benchmarks não informaram")
    for parametro in ("training_loop", "negative_sampler", "optimizer", "evaluator", "stopper",
                      "filter_validation_when_testing"):
        if parametro in padroes:
            print(f"   {parametro}: {padroes[parametro].default!r}")
    avaliador = evaluator_resolver.make(None)
    print(f"   avaliador padrão: {nome(avaliador)} (filtered={getattr(avaliador, 'filtered', '?')})")

    triplas = ler_triplas(PASTA_XML)
    fabrica = TriplesFactory.from_labeled_triples(triples=triplas)
    treino, teste = fabrica.split([0.8, 0.2], random_state=42)
    print(f"\nGrafo: {fabrica.num_triples} triplas, {fabrica.num_entities} entidades,"
          f" {fabrica.num_relations} tipos de relação | treino {treino.num_triples}, teste {teste.num_triples}")

    for nome_modelo in MODELOS:
        print(f"\n== {nome_modelo} (mesmos argumentos do benchmark, mas 1 época em vez de 10)")
        resultado = pipeline(
            training=treino,
            testing=teste,
            model=nome_modelo,
            model_kwargs=dict(embedding_dim=64),
            training_kwargs=dict(num_epochs=1, batch_size=256, use_tqdm=False),
            evaluation_kwargs=dict(batch_size=256, use_tqdm=False),
            random_seed=0,
            device="cpu",
        )
        descrever(resultado, treino)


if __name__ == "__main__":
    main()
