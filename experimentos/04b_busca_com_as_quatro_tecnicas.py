"""
Experimento 4b: a busca do experimento 4, com as quatro técnicas.

O experimento 4 mediu, só com o RotatE, se os vetores ajudam a recuperar ligações que o grafo não tem. Aqui a mesma
busca (a parte 1 daquele experimento, sem modelo de linguagem) é repetida com TransE, ComplEx, RotatE e Node2Vec, para
saber qual técnica serve melhor a esse uso.

As perguntas e as respostas escondidas são as do experimento 4: o grafo consultado é o de treino (80% das ligações),
e cada pergunta é um conceito com pelo menos uma ligação de associação escondida no teste. O script confere que as
fontes sem vetores dão os mesmos números do experimento 4.

Fontes de sugestões (5 conceitos por pergunta, fora o próprio termo e o que já está ligado a ele):
  popularidade  os conceitos com mais ligações de associação no grafo                          (sem vetores)
  dois saltos   os conceitos com mais vizinhos em comum com o termo                            (sem vetores)
  parecidos     os de vetor mais parecido (cosseno), sem o corte de 0,5 do protótipo           (as quatro técnicas)
  prováveis     as ligações que o modelo pontua como mais prováveis, nos dois sentidos         (TransE, ComplEx, RotatE)
O Node2Vec não tem "prováveis", porque não distingue os tipos de ligação.

Cada técnica é treinada 5 vezes (sementes 0 a 4): TransE, ComplEx e RotatE com a configuração escolhida no
experimento 1, e o Node2Vec com os parâmetros de src/embeddings.py (a implementação do experimento 7).

Medida: em quantas perguntas pelo menos uma resposta escondida aparece entre as 5 sugestões. Sai a média e o desvio das
5 sementes e, para cada fonte com vetores, a diferença para a melhor fonte sem vetores, com intervalo de 95% sorteando
perguntas (cada pergunta entra com a média das sementes).

Saída (não versionada):
  results/experimento4b_busca.json

Uso, no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python experimentos/04b_busca_com_as_quatro_tecnicas.py --rapido   # testa o código: 1 semente, 20 épocas
    python experimentos/04b_busca_com_as_quatro_tecnicas.py            # continua de onde parou, se for interrompido
"""
import argparse
import collections
import importlib.util
import json
import logging
import os
import statistics
import sys
import time
import warnings

import numpy as np
import torch
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento4b_busca.json")
EXPERIMENTO4 = os.path.join(RAIZ, "results", "experimento4_rag.json")
EXPERIMENTO1 = os.path.join(RAIZ, "results", "experimento1_ampliado_resultados.json")
KGE = ("TransE", "ComplEx", "RotatE")
SEM_VETORES = ("popularidade", "dois saltos")
SEMENTES = (0, 1, 2, 3, 4)
REAMOSTRAS = 2000


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Perguntas, grafo e fontes sem vetores: os do experimento 4; Node2Vec: o do experimento 7
rag = carregar("rag", "04_avaliacao_rag.py")
sugestao = carregar("sugestao", "07_sugestao_de_diagnosticos.py")
retreino, agrupamento = rag.retreino, rag.agrupamento


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def preparar():
    """O grafo de treino, as perguntas, as respostas escondidas e o que não pode ser sugerido, como no experimento 4."""
    fabrica = TriplesFactory.from_labeled_triples(triples=retreino.ler_triplas(retreino.PASTA_XML))
    treino, validacao, teste = fabrica.split(list(retreino.DIVISAO), random_state=retreino.SEMENTE_DIVISAO)
    relacao = treino.relation_to_id[rag.RELACAO]
    _, vizinhos = rag.grafo(treino.mapped_triples)
    texto = {i: rag.normalizar(rag.texto(r)) for r, i in treino.entity_to_id.items()}
    ids_por_texto = collections.defaultdict(set)
    for i, t in texto.items():
        ids_por_texto[t].add(i)

    def textos_conhecidos(e):
        return {texto[c] for c in vizinhos[e]} | {texto[e]}

    def excluidos(e):
        return set().union(*(ids_por_texto[t] for t in textos_conhecidos(e))) | vizinhos[e] | {e}

    escondidas = collections.defaultdict(set)
    for h, r, t in teste.mapped_triples.tolist():
        if r != relacao or h == t or t in vizinhos[h]:
            continue
        if texto[t] not in textos_conhecidos(h):
            escondidas[h].add(t)
        if texto[h] not in textos_conhecidos(t):
            escondidas[t].add(h)
    perguntas = sorted(escondidas)
    return treino, validacao, relacao, vizinhos, escondidas, perguntas, {e: excluidos(e) for e in perguntas}


def acertos(sugestoes, escondidas, perguntas):
    """Para cada pergunta, 1 se alguma resposta escondida está entre as sugestões."""
    return [int(bool(set(sugestoes[e]) & escondidas[e])) for e in perguntas]


def sem_vetores(treino, relacao, vizinhos, perguntas, fora):
    grau = collections.Counter()
    for h, r, t in treino.mapped_triples.tolist():
        if r == relacao:
            grau[h] += 1
            grau[t] += 1
    populares = sorted(range(treino.num_entities), key=lambda e: (-grau[e], e))
    saida = {"popularidade": {}, "dois saltos": {}}
    for e in perguntas:
        escolhidos = []
        for c in populares:
            if c not in fora[e]:
                escolhidos.append(c)
                if len(escolhidos) == rag.SUGESTOES:
                    break
        saida["popularidade"][e] = escolhidos
        comuns = collections.Counter(w for v in vizinhos[e] for w in vizinhos[v] if w not in fora[e])
        saida["dois saltos"][e] = sorted(comuns, key=lambda c: (-comuns[c], -len(vizinhos[c]), c))[:rag.SUGESTOES]
    return saida


def parecidos(x, perguntas, fora):
    return {e: rag.melhores(x @ x[e], fora[e]) for e in perguntas}


def provaveis(modelo, relacao, perguntas, fora):
    saida = {}
    modelo.eval()
    with torch.no_grad():
        for inicio in range(0, len(perguntas), 256):
            bloco = torch.as_tensor(perguntas[inicio:inicio + 256])
            rel = torch.full_like(bloco, relacao)
            escores = torch.maximum(modelo.score_t(torch.stack([bloco, rel], dim=1)),   # termo --> candidato
                                    modelo.score_h(torch.stack([rel, bloco], dim=1))).numpy()  # candidato --> termo
            for i, e in enumerate(bloco.tolist()):
                saida[e] = rag.melhores(escores[i], fora[e])
    return saida


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--rapido", action="store_true", help="1 semente e 20 épocas: testa o código")
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)

    treino, validacao, relacao, vizinhos, escondidas, perguntas, fora = preparar()
    print(f"{len(perguntas)} perguntas, as mesmas do experimento 4")
    fontes = {f: acertos(s, escondidas, perguntas) for f, s in sem_vetores(treino, relacao, vizinhos, perguntas, fora).items()}
    antes = ler_json(EXPERIMENTO4)["recuperacao"]
    for f in SEM_VETORES:
        assert abs(statistics.fmean(fontes[f]) - antes[f]["perguntas_com_acerto"]) < 1e-9, f"{f} não deu o número do experimento 4"

    saida = SAIDA_JSON.replace(".json", "_teste.json") if args.rapido else SAIDA_JSON
    dados = ler_json(saida) if os.path.isfile(saida) else {"perguntas": len(perguntas), "sem_vetores": fontes, "execucoes": {}}
    configuracoes = ler_json(EXPERIMENTO1)["resumo"]
    triplas_de_treino = {tuple(t) for t in treino.triples}
    for semente in (SEMENTES[:1] if args.rapido else SEMENTES):
        for tecnica in (*KGE, "Node2Vec"):
            chave = f"{tecnica}_semente{semente}"
            if chave in dados["execucoes"]:
                continue
            inicio = time.time()
            if tecnica == "Node2Vec":
                indice, vetores = sugestao.treinar_node2vec(triplas_de_treino, semente, 1 if args.rapido else sugestao.N2V_EPOCAS)
                x = vetores[[indice[e] for e in sorted(treino.entity_to_id, key=treino.entity_to_id.get)]]
                execucao = {"parecidos": acertos(parecidos(x, perguntas, fora), escondidas, perguntas)}
            else:
                c = configuracoes[tecnica]
                modelo, _ = retreino.treinar(tecnica, c["dimensao"], c["configuracao"]["perda"], c["configuracao"]["negativos"],
                                             c["configuracao"]["taxa"], semente, treino, validacao,
                                             20 if args.rapido else retreino.MAX_EPOCAS)
                x = agrupamento.preparar(modelo.entity_representations[0](indices=None).detach().cpu().numpy())
                execucao = {"parecidos": acertos(parecidos(x, perguntas, fora), escondidas, perguntas),
                            "prováveis": acertos(provaveis(modelo, relacao, perguntas, fora), escondidas, perguntas)}
            dados["execucoes"][chave] = execucao
            retreino.salvar_json(dados, saida)
            print(f"   {chave:22s}" + "".join(f"  {f}: {statistics.fmean(a):.1%}" for f, a in execucao.items())
                  + f"  ({time.time() - inicio:.0f} s)", flush=True)

    # A tabela: média e desvio das sementes, e a diferença para a melhor fonte sem vetores
    melhor = max(SEM_VETORES, key=lambda f: statistics.fmean(fontes[f]))
    base = np.array(fontes[melhor], dtype=float)
    sorteios = np.random.default_rng(0).integers(0, len(perguntas), (REAMOSTRAS, len(perguntas)))
    tabela = {f: {"perguntas_com_acerto": statistics.fmean(fontes[f])} for f in SEM_VETORES}
    for tecnica in (*KGE, "Node2Vec"):
        execucoes = [v for k, v in dados["execucoes"].items() if k.startswith(f"{tecnica}_semente")]
        for fonte in execucoes[0]:
            por_semente = np.array([e[fonte] for e in execucoes], dtype=float)
            diferenca = por_semente.mean(axis=0) - base
            baixo, alto = np.percentile(diferenca[sorteios].mean(axis=1), [2.5, 97.5])
            taxas = por_semente.mean(axis=1)
            tabela[f"{fonte}, {tecnica}"] = {
                "perguntas_com_acerto": float(taxas.mean()), "desvio": float(taxas.std()), "sementes": len(execucoes),
                "diferenca_para_a_melhor_sem_vetores": float(diferenca.mean()), "intervalo_da_diferenca": [float(baixo), float(alto)]}
    dados["tabela"], dados["melhor_sem_vetores"] = tabela, melhor
    retreino.salvar_json(dados, saida)

    print(f"\n== A resposta escondida aparece entre as {rag.SUGESTOES} sugestões? ({len(perguntas)} perguntas)")
    print(f"   {'fonte':24s}{'perguntas com acerto':>22s}{'desvio':>9s}{f'diferença para {melhor} (95%)':>42s}")
    for fonte, r in sorted(tabela.items(), key=lambda x: -x[1]["perguntas_com_acerto"]):
        if fonte in SEM_VETORES:
            print(f"   {fonte + ' (sem vetores)':24s}{r['perguntas_com_acerto']:22.1%}")
        else:
            d, (baixo, alto) = r["diferenca_para_a_melhor_sem_vetores"], r["intervalo_da_diferenca"]
            intervalo = f"{100 * d:+.1f} ({100 * baixo:+.1f} a {100 * alto:+.1f}) pontos"
            print(f"   {fonte:24s}{r['perguntas_com_acerto']:22.1%}{r['desvio']:9.1%}{intervalo:>42s}")
    print(f"\nResultados em {os.path.relpath(saida, RAIZ)}")


if __name__ == "__main__":
    main()
