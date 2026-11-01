"""
Experimento 4: o grafo e os vetores ajudam o modelo de linguagem a responder?

Segunda parte do objetivo do TCC: aplicar a técnica de melhor desempenho
(o RotatE) num protótipo de RAG e avaliar se a representação vetorial
contribui para a recuperação de contexto.

Cenário. Um pesquisador pergunta ao sistema: "No corpus SemClinBr, quais
conceitos aparecem associados a X?". Para saber se o sistema traz
informação nova, e não só a que já está no grafo, parte das ligações fica
escondida: o grafo consultado é só o de treino (a mesma divisão dos
experimentos anteriores), e as ligações de associação do teste que ele não
contém são as respostas escondidas. Uma ligação escondida só conta se
trouxer um texto novo: muitos conceitos aparecem com o mesmo texto e
categorias diferentes, e o mesmo texto já ligado ao termo não é informação
nova. Há uma pergunta para cada entidade com pelo menos uma ligação
escondida.

Fontes de sugestões para cada pergunta (5 conceitos cada, fora o próprio
termo, o que já está ligado a ele no grafo e os conceitos com o mesmo
texto de um desses):
  popularidade  os conceitos com mais ligações de associação no grafo
  dois saltos   os conceitos com mais vizinhos em comum com o termo
  parecidos     os de vetor mais parecido (cosseno), como no protótipo:
                só entram os de similaridade 0,5 ou mais (na parte 1,
                aparece também a versão sem esse corte)
  prováveis     as ligações que o RotatE pontua como mais prováveis
As duas primeiras não usam vetores e servem de comparação.

Parte 1, recuperação (todas as perguntas, sem modelo de linguagem): em
quantas perguntas pelo menos uma resposta escondida aparece entre as 5
sugestões de cada fonte.

Parte 2, resposta (amostra de perguntas, com o modelo de linguagem do
protótipo, llama3.2 via Ollama, temperatura 0): cada pergunta é feita com
quatro contextos (nenhum; grafo; grafo + parecidos; grafo + prováveis). O
contexto "grafo" traz até 10 ligações diretas do termo, como a consulta
Cypher do protótipo. A resposta deve listar até 5 conceitos, e cada um é
conferido com o corpus inteiro:
  acerto      pelo menos um conceito listado está ligado ao termo no corpus
  precisão    fração dos conceitos listados (somando todas as respostas)
              que estão ligados ao termo
  escondido   pelo menos um conceito listado é uma resposta escondida; ao
              lado, em quantas perguntas o contexto trazia uma delas
  fora do contexto  fração dos conceitos listados que não estavam no
              contexto dado ao modelo, isto é, que ele inventou

O RotatE é retreinado com a configuração escolhida no experimento 1 e a
semente 0 (os vetores são conferidos com os salvos naquele experimento).
As ligações vêm dos XMLs, como nos outros experimentos, então não é
preciso o Neo4j rodando: as ligações diretas são as mesmas que a consulta
Cypher do protótipo devolveria. Simplificações: o termo da pergunta já é
dado (o protótipo o reconhece por correspondência de texto), e um conceito
listado conta como certo quando o texto dele, sem a categoria entre
parênteses, é igual ao de uma entidade ligada ao termo, sem diferenciar
maiúsculas, acentos e pontuação. Quando o modelo copia a linha do contexto
("A --[relação]--> B"), vale o conceito do outro lado da ligação.

Saída (não versionada):
  results/experimento4_rag.json

Uso, no ambiente conda do projeto (no WSL), com o Ollama rodando e o
modelo baixado (ollama pull llama3.2):
    conda activate tcc_graphs
    python experimentos/04_avaliacao_rag.py --perguntas 2   # teste rápido
    python experimentos/04_avaliacao_rag.py                 # 100 perguntas
    python experimentos/04_avaliacao_rag.py --reavaliar     # só confere de novo as respostas salvas
"""
import argparse
import collections
import importlib.util
import json
import logging
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import warnings

import numpy as np
import requests
import torch
from pykeen.triples import TriplesFactory

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento4_rag.json")
RESULTADOS_EXPERIMENTO1 = os.path.join(RAIZ, "results", "experimento1_ampliado_resultados.json")
VETORES_EXPERIMENTO1 = os.path.join(RAIZ, "results", "experimento1_ampliado_vetores", "RotatE_semente0.npz")
RELACAO = "associated_with"
SUGESTOES, LIGACOES_NO_CONTEXTO, CONCEITOS_NA_RESPOSTA = 5, 10, 5
SIMILARIDADE_MINIMA = 0.5  # corte do protótipo (rag_retriever.py) para um vetor parecido entrar no contexto
SEMENTE = 0
FONTES = ("popularidade", "dois saltos", "parecidos", "parecidos sem corte", "prováveis")
CONTEXTOS = ("nenhum", "grafo", "grafo + parecidos", "grafo + prováveis")

PROMPT_SEM_CONTEXTO = """Você é um assistente de apoio à pesquisa clínica. Responda à pergunta com base \
no seu conhecimento. Responda apenas com até 5 conceitos, um por linha, sem explicações.

Pergunta: {pergunta}

Resposta:"""

# Mesmas instruções do protótipo (rag_llm.py), com a terceira fonte e o formato da resposta.
PROMPT_COM_CONTEXTO = """Você é um assistente de apoio à pesquisa clínica. Responda à pergunta usando \
SOMENTE as informações do contexto abaixo, extraído de um Grafo de Conhecimento Clínico construído a \
partir do corpus SemClinBr. Se o contexto não for suficiente, responda NENHUM.

O contexto pode ter três tipos de linha, com confiabilidade diferente:
- "A --[relação]--> B": anotação real do corpus; evidência forte.
- "A é semanticamente próximo de B [similaridade=X]": vem de um espaço vetorial aproximado; evidência fraca.
- "A --[relação]--> B [ligação provável]": não está anotada no corpus; é uma previsão do modelo; evidência fraca.

Responda apenas com até 5 conceitos, um por linha, escritos como aparecem no contexto, sem explicações.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Triplas, divisão, treino e preparação dos vetores: as mesmas dos experimentos 1 e 3
retreino = carregar("retreino", "01_retreino_kge.py")
agrupamento = carregar("agrupamento", "03_agrupamento.py")


def texto(rotulo):
    return rotulo.split("|", 1)[0]


def categoria(rotulo):
    return rotulo.split("|", 1)[1] if "|" in rotulo else ""


def normalizar(t):
    """Maiúsculas, sem acentos e com a pontuação trocada por espaço."""
    t = unicodedata.normalize("NFKD", t.upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", t).split())


def grafo(triplas):
    """Ligações (relação, sai da entidade?, outra entidade) e vizinhos de cada entidade."""
    ligacoes, vizinhos = collections.defaultdict(list), collections.defaultdict(set)
    for h, r, t in triplas.tolist():
        ligacoes[h].append((r, True, t))
        ligacoes[t].append((r, False, h))
        vizinhos[h].add(t)
        vizinhos[t].add(h)
    return ligacoes, vizinhos


def melhores(escores, fora, k=SUGESTOES):
    """Os k índices de maior escore, fora os excluídos."""
    escores = escores.copy()
    escores[list(fora)] = -np.inf
    topo = np.argpartition(-escores, k)[:k]
    return [int(c) for c in topo[np.argsort(-escores[topo])]]


def chance_ao_acaso(escondidas, candidatos, k=SUGESTOES):
    """Probabilidade de k sugestões sorteadas incluírem pelo menos uma das escondidas."""
    return 1 - math.prod((candidatos - escondidas - i) / (candidatos - i) for i in range(k))


def perguntar(prompt, modelo, host):
    resposta = requests.post(
        f"{host}/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False, "keep_alive": "30m",
              "options": {"temperature": 0, "seed": SEMENTE, "num_predict": 150}},
        timeout=600,
    )
    resposta.raise_for_status()
    return resposta.json().get("response", "").strip()


def conceitos(resposta, termo):
    """Conceitos listados na resposta: um por linha ou separados por vírgula, sem marcadores de
    lista, categoria entre parênteses e observações entre colchetes. Quando o modelo copia a linha
    do contexto ("A --[relação]--> B"), fica o lado que não é o termo da pergunta."""
    termo = normalizar(termo)
    itens = []
    for linha in resposta.splitlines():
        linha = re.sub(r"^\s*(?:[-*•]+|\d+\s*[.)-])\s*", "", linha).strip().strip("*").strip()
        # pula linhas vazias, títulos ("Conceitos:") e a resposta NENHUM
        if not normalizar(linha) or linha.endswith(":") or normalizar(linha).startswith("NENHUM"):
            continue
        linha = re.sub(r"--\[[^\]]*\]-->", " --> ", linha)
        linha = re.sub(r"\s+(?:é|e) semanticamente pr[oó]ximo de\s+", " --> ", linha)
        linha = re.sub(r"\s+no espa[cç]o de embeddings", " ", linha)
        linha = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", linha)
        for parte in re.split(r"(?<=[^\d\s]),\s+|;\s*", linha):  # não separa números como "4 , 01"
            lados = [lado.strip() for lado in parte.split("-->") if normalizar(lado)]
            if len(lados) > 1:
                lados = [lado for lado in lados if normalizar(lado) != termo][:1]
            itens.extend(lado for lado in lados if normalizar(lado) not in {normalizar(i) for i in itens})
    return itens[:CONCEITOS_NA_RESPOSTA]


def avaliar_resposta(resposta, linhas, termo, verdadeiros, alvos_escondidos):
    """Confere os conceitos listados com as ligações do corpus e com o contexto dado ao modelo."""
    itens = conceitos(resposta, termo)
    corretos = [normalizar(item) in verdadeiros for item in itens]
    no_contexto, escondido_no_contexto = None, None
    if linhas is not None:
        do_contexto = {normalizar(c) for linha in linhas for c in conceitos(linha, termo)}
        no_contexto = [normalizar(item) in do_contexto for item in itens]
        escondido_no_contexto = bool(do_contexto & alvos_escondidos)
    return {"conceitos": itens, "corretos": corretos, "no_contexto": no_contexto,
            "acerto": any(corretos), "escondido": any(normalizar(item) in alvos_escondidos for item in itens),
            "escondido_no_contexto": escondido_no_contexto}


def resumir_respostas(detalhes):
    respostas = {}
    for nome in CONTEXTOS:
        validas = [d["contextos"][nome] for d in detalhes if "erro" not in d["contextos"][nome]]
        # precisão e "fora do contexto" somam os conceitos de todas as respostas
        corretos = [ok for r in validas for ok in r["corretos"]]
        no_contexto = [ok for r in validas if r["no_contexto"] is not None for ok in r["no_contexto"]]
        com_contexto = [r for r in validas if r["escondido_no_contexto"] is not None]
        respostas[nome] = {
            "perguntas": len(validas),
            "acerto": statistics.fmean(r["acerto"] for r in validas) if validas else None,
            "precisao": statistics.fmean(corretos) if corretos else None,
            "escondido": statistics.fmean(r["escondido"] for r in validas) if validas else None,
            "escondido_no_contexto": (statistics.fmean(r["escondido_no_contexto"] for r in com_contexto)
                                      if com_contexto else None),
            "fora_do_contexto": statistics.fmean(not ok for ok in no_contexto) if no_contexto else None,
            "sem_resposta": statistics.fmean(not r["conceitos"] for r in validas) if validas else None,
        }
    return respostas


def imprimir_respostas(respostas, modelo, perguntas):
    def pct(v):
        return "    -" if v is None else f"{100 * v:5.1f}%"

    print(f"\n== Parte 2: respostas do {modelo} ({perguntas} perguntas)")
    print(f"   {'contexto':20s}{'acerto':>9s}{'precisão':>10s}{'escondido':>11s}{'(no contexto)':>15s}"
          f"{'fora do contexto':>18s}{'sem resposta':>14s}")
    for nome, r in respostas.items():
        print(f"   {nome:20s}{pct(r['acerto']):>9s}{pct(r['precisao']):>10s}{pct(r['escondido']):>11s}"
              f"{pct(r['escondido_no_contexto']):>15s}{pct(r['fora_do_contexto']):>18s}{pct(r['sem_resposta']):>14s}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--perguntas", type=int, default=100,
                            help="perguntas feitas ao modelo de linguagem (0: só a parte 1)")
    argumentos.add_argument("--modelo", default="llama3.2", help="modelo do Ollama (padrão: o do protótipo)")
    argumentos.add_argument("--host", default="http://localhost:11434", help="endereço do Ollama")
    argumentos.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                            help="núcleos de CPU usados pelo PyTorch (padrão: metade, como no experimento 1)")
    argumentos.add_argument("--reavaliar", action="store_true",
                            help="só confere de novo as respostas já salvas, sem treinar nem chamar o modelo")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(args.threads)

    info_modelo = None
    if args.perguntas > 0 and not args.reavaliar:
        try:
            instalados = requests.get(f"{args.host}/api/tags", timeout=10).json().get("models", [])
        except requests.exceptions.RequestException as erro:
            sys.exit(f"Não foi possível falar com o Ollama em {args.host} ({erro}). Ele está rodando?")
        info_modelo = next((m for m in instalados if args.modelo in (m["name"], m["name"].split(":")[0])), None)
        if info_modelo is None:
            sys.exit(f"O modelo '{args.modelo}' não está no Ollama. Rode: ollama pull {args.modelo}")

    fabrica = TriplesFactory.from_labeled_triples(triples=retreino.ler_triplas(retreino.PASTA_XML))
    treino, validacao, teste = fabrica.split(list(retreino.DIVISAO), random_state=retreino.SEMENTE_DIVISAO)
    rotulos = {i: r for r, i in treino.entity_to_id.items()}
    nomes_relacao = {i: r for r, i in treino.relation_to_id.items()}
    relacao = treino.relation_to_id[RELACAO]
    n = treino.num_entities

    ligacoes, vizinhos = grafo(treino.mapped_triples)
    _, vizinhos_corpus = grafo(torch.cat([treino.mapped_triples, validacao.mapped_triples, teste.mapped_triples]))
    texto_normalizado = {i: normalizar(texto(r)) for i, r in rotulos.items()}
    ids_por_texto = collections.defaultdict(set)
    for i, t in texto_normalizado.items():
        ids_por_texto[t].add(i)

    def textos_conhecidos(e):
        """Textos do termo e das entidades já ligadas a ele no grafo de treino."""
        return {texto_normalizado[c] for c in vizinhos[e]} | {texto_normalizado[e]}

    def excluidos(e):
        """O termo, o que já está ligado a ele e as entidades com o mesmo texto de um desses
        (o mesmo conceito com outra categoria não é informação nova)."""
        return set().union(*(ids_por_texto[t] for t in textos_conhecidos(e))) | vizinhos[e] | {e}

    escondidas, ja_no_grafo, mesmo_texto = collections.defaultdict(set), 0, 0
    for h, r, t in teste.mapped_triples.tolist():
        if r != relacao or h == t:
            continue
        if t in vizinhos[h]:  # o par já está ligado no grafo de treino (no outro sentido ou pela outra relação)
            ja_no_grafo += 1
            continue
        nova = False
        if texto_normalizado[t] not in textos_conhecidos(h):
            escondidas[h].add(t)
            nova = True
        if texto_normalizado[h] not in textos_conhecidos(t):
            escondidas[t].add(h)
            nova = True
        mesmo_texto += not nova
    perguntas = sorted(escondidas)
    print(f"Ligações de associação no teste: {sum(1 for _, r, _ in teste.mapped_triples.tolist() if r == relacao)};"
          f" ficaram de fora {ja_no_grafo} que já estavam no grafo de treino e {mesmo_texto} que só repetiam"
          f" o texto de um conceito já ligado.")
    print(f"Perguntas: {len(perguntas)} entidades com pelo menos uma ligação escondida.")

    def alvos(e):
        """Textos das entidades ligadas ao termo no corpus, e só os das ligações escondidas."""
        verdadeiros = {texto_normalizado[c] for c in vizinhos_corpus[e]} - {""}
        return verdadeiros, {texto_normalizado[c] for c in escondidas[e]} - {""}

    if args.reavaliar:
        with open(SAIDA_JSON, encoding="utf-8") as f:
            dados = json.load(f)
        for registro in dados["detalhes"]:
            verdadeiros, escondidos = alvos(treino.entity_to_id[registro["entidade"]])
            for c in registro["contextos"].values():
                if "erro" not in c:
                    c.update(avaliar_resposta(c["resposta"], c["linhas"], texto(registro["entidade"]),
                                              verdadeiros, escondidos))
        dados["respostas"] = resumir_respostas(dados["detalhes"])
        retreino.salvar_json(dados, SAIDA_JSON)
        imprimir_respostas(dados["respostas"], dados["protocolo"]["modelo_de_linguagem"]["nome"], len(dados["detalhes"]))
        print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")
        return

    with open(RESULTADOS_EXPERIMENTO1, encoding="utf-8") as f:
        resumo_rotate = json.load(f)["resumo"]["RotatE"]
    config = resumo_rotate["configuracao"]
    print(f"\nTreinando o RotatE (perda {config['perda']}, {config['negativos']} negativos, taxa {config['taxa']},"
          f" semente {SEMENTE})...")
    modelo, info = retreino.treinar("RotatE", resumo_rotate["dimensao"], config["perda"], config["negativos"],
                                    config["taxa"], SEMENTE, treino, validacao, retreino.MAX_EPOCAS)
    vetores = modelo.entity_representations[0](indices=None).detach().cpu().numpy()
    iguais = None
    if os.path.isfile(VETORES_EXPERIMENTO1):
        salvos = np.load(VETORES_EXPERIMENTO1)["vetores"]
        iguais = bool(salvos.shape == vetores.shape and np.allclose(salvos, vetores, atol=1e-5))
    print(f"   melhor época {info['melhor_epoca']}, {info['segundos']} s;"
          f" vetores iguais aos salvos no experimento 1: {'sim' if iguais else 'não' if iguais is False else '?'}")
    x = agrupamento.preparar(vetores)

    # Sugestões de cada fonte
    grau_relacao = collections.Counter()
    for h, r, t in treino.mapped_triples.tolist():
        if r == relacao:
            grau_relacao[h] += 1
            grau_relacao[t] += 1
    ranking_popularidade = sorted(range(n), key=lambda e: (-grau_relacao[e], e))
    sugestoes = {fonte: {} for fonte in FONTES}
    similaridade, sai_do_termo = {}, {}
    for e in perguntas:
        fora = excluidos(e)
        populares = []
        for c in ranking_popularidade:
            if c not in fora:
                populares.append(c)
                if len(populares) == SUGESTOES:
                    break
        sugestoes["popularidade"][e] = populares
        comuns = collections.Counter(w for v in vizinhos[e] for w in vizinhos[v] if w not in fora)
        sugestoes["dois saltos"][e] = sorted(comuns, key=lambda c: (-comuns[c], -len(vizinhos[c]), c))[:SUGESTOES]
        cossenos = x @ x[e]
        sugestoes["parecidos sem corte"][e] = melhores(cossenos, fora)
        sugestoes["parecidos"][e] = [c for c in sugestoes["parecidos sem corte"][e] if cossenos[c] >= SIMILARIDADE_MINIMA]
        similaridade[e] = {c: float(cossenos[c]) for c in sugestoes["parecidos sem corte"][e]}
    modelo.eval()
    with torch.no_grad():
        for inicio in range(0, len(perguntas), 256):
            bloco = torch.as_tensor(perguntas[inicio:inicio + 256])
            rel = torch.full_like(bloco, relacao)
            como_origem = modelo.score_t(torch.stack([bloco, rel], dim=1))   # termo --> candidato
            como_destino = modelo.score_h(torch.stack([rel, bloco], dim=1))  # candidato --> termo
            escores = torch.maximum(como_origem, como_destino).numpy()
            sai = (como_origem >= como_destino).numpy()
            for i, e in enumerate(bloco.tolist()):
                sugestoes["prováveis"][e] = melhores(escores[i], excluidos(e))
                sai_do_termo[e] = {c: bool(sai[i, c]) for c in sugestoes["prováveis"][e]}

    print(f"\n== Parte 1: a ligação escondida aparece entre as {SUGESTOES} sugestões? ({len(perguntas)} perguntas)")
    recuperacao = {"acaso": {
        "perguntas_com_acerto": statistics.fmean(
            chance_ao_acaso(len(escondidas[e]), n - len(excluidos(e))) for e in perguntas),
        "sugestoes_corretas": statistics.fmean(len(escondidas[e]) / (n - len(excluidos(e))) for e in perguntas),
        "sugestoes_por_pergunta": SUGESTOES,
    }}
    for fonte in FONTES:
        acertos = [len(set(sugestoes[fonte][e]) & escondidas[e]) for e in perguntas]
        total = sum(len(sugestoes[fonte][e]) for e in perguntas)
        recuperacao[fonte] = {
            "perguntas_com_acerto": statistics.fmean(a > 0 for a in acertos),
            "sugestoes_corretas": sum(acertos) / total if total else None,
            "sugestoes_por_pergunta": total / len(perguntas),
        }
    print(f"   {'fonte':30s}{'perguntas com acerto':>22s}{'sugestões corretas':>20s}{'sugestões por pergunta':>24s}")
    for fonte, r in recuperacao.items():
        nome = f"{fonte} (vetores)" if fonte.startswith(("parecidos", "prováveis")) else fonte
        corretas = "-" if r["sugestoes_corretas"] is None else f"{100 * r['sugestoes_corretas']:.1f}%"
        print(f"   {nome:30s}{100 * r['perguntas_com_acerto']:21.1f}%{corretas:>20s}{r['sugestoes_por_pergunta']:24.1f}")

    dados = {
        "versoes": {"pykeen": retreino.pykeen.get_version(), "torch": torch.__version__},
        "protocolo": {"relacao": RELACAO, "sugestoes": SUGESTOES, "ligacoes_no_contexto": LIGACOES_NO_CONTEXTO,
                      "conceitos_na_resposta": CONCEITOS_NA_RESPOSTA, "semente": SEMENTE,
                      "rotate": {"dimensao": resumo_rotate["dimensao"], **config, **info,
                                 "vetores_iguais_ao_experimento1": iguais},
                      "modelo_de_linguagem": None if info_modelo is None else {
                          "nome": info_modelo["name"], "digest": info_modelo.get("digest"),
                          **{k: info_modelo.get("details", {}).get(k) for k in ("parameter_size", "quantization_level")}},
                      "prompt_sem_contexto": PROMPT_SEM_CONTEXTO, "prompt_com_contexto": PROMPT_COM_CONTEXTO},
        "ligacoes_ja_no_grafo": ja_no_grafo, "ligacoes_com_texto_repetido": mesmo_texto,
        "perguntas_parte1": len(perguntas),
        "recuperacao": recuperacao, "respostas": None, "detalhes": [],
    }
    retreino.salvar_json(dados, SAIDA_JSON)
    if args.perguntas <= 0:
        print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")
        return

    # Parte 2: amostra de perguntas com termos legíveis (pelo menos 4 caracteres e 3 letras)
    elegiveis = [e for e in perguntas if len(normalizar(texto(rotulos[e])).replace(" ", "")) >= 4
                 and sum(ch.isalpha() for ch in texto(rotulos[e])) >= 3]
    rng = np.random.default_rng(SEMENTE)
    amostra = sorted(int(e) for e in rng.choice(elegiveis, size=min(args.perguntas, len(elegiveis)), replace=False))

    def linhas_grafo(e):
        ordem = sorted(ligacoes[e], key=lambda l: (-len(vizinhos[l[2]]), rotulos[l[2]]))[:LIGACOES_NO_CONTEXTO]
        return [f"{texto(rotulos[e])} --[{nomes_relacao[r]}]--> {texto(rotulos[c])} ({categoria(rotulos[c])})" if sai
                else f"{texto(rotulos[c])} ({categoria(rotulos[c])}) --[{nomes_relacao[r]}]--> {texto(rotulos[e])}"
                for r, sai, c in ordem]

    def linhas_parecidos(e):
        return [f"{texto(rotulos[e])} é semanticamente próximo de {texto(rotulos[c])} ({categoria(rotulos[c])})"
                f" no espaço de embeddings [similaridade={similaridade[e][c]:.2f}]" for c in sugestoes["parecidos"][e]]

    def linhas_provaveis(e):
        return [f"{texto(rotulos[e])} --[{RELACAO}]--> {texto(rotulos[c])} ({categoria(rotulos[c])}) [ligação provável]"
                if sai_do_termo[e][c] else
                f"{texto(rotulos[c])} ({categoria(rotulos[c])}) --[{RELACAO}]--> {texto(rotulos[e])} [ligação provável]"
                for c in sugestoes["prováveis"][e]]

    print(f"\n== Parte 2: {len(amostra)} perguntas ao {info_modelo['name']}, com {len(CONTEXTOS)} contextos cada")
    inicio_parte2 = time.time()
    for numero, e in enumerate(amostra, 1):
        pergunta = f'No corpus SemClinBr, quais conceitos aparecem associados a "{texto(rotulos[e])}"?'
        verdadeiros, alvos_escondidos = alvos(e)
        contextos = {"nenhum": None, "grafo": linhas_grafo(e),
                     "grafo + parecidos": linhas_grafo(e) + linhas_parecidos(e),
                     "grafo + prováveis": linhas_grafo(e) + linhas_provaveis(e)}
        registro = {"entidade": rotulos[e], "pergunta": pergunta,
                    "escondidas": sorted(rotulos[c] for c in escondidas[e]), "contextos": {}}
        marcas = []
        for nome, linhas in contextos.items():
            prompt = (PROMPT_SEM_CONTEXTO.format(pergunta=pergunta) if linhas is None else
                      PROMPT_COM_CONTEXTO.format(contexto="\n".join(linhas), pergunta=pergunta))
            comeco = time.time()
            try:
                resposta = perguntar(prompt, info_modelo["name"], args.host)
            except requests.exceptions.RequestException as erro:  # registra a falha em vez de contar como erro do modelo
                registro["contextos"][nome] = {"linhas": linhas, "erro": repr(erro)}
                marcas.append(f"{nome}: FALHOU")
                continue
            r = {"linhas": linhas, "resposta": resposta, "segundos": round(time.time() - comeco, 1),
                 **avaliar_resposta(resposta, linhas, texto(rotulos[e]), verdadeiros, alvos_escondidos)}
            registro["contextos"][nome] = r
            marcas.append(f"{nome}: {sum(r['corretos'])}/{len(r['conceitos'])}{' +escondido' if r['escondido'] else ''}")
        dados["detalhes"].append(registro)
        retreino.salvar_json(dados, SAIDA_JSON)
        decorrido = time.time() - inicio_parte2
        print(f"   [{numero}/{len(amostra)}] {texto(rotulos[e])[:40]:40s} | " + " | ".join(marcas)
              + f" ({decorrido / numero:.0f} s por pergunta)")

    dados["respostas"] = resumir_respostas(dados["detalhes"])
    retreino.salvar_json(dados, SAIDA_JSON)
    imprimir_respostas(dados["respostas"], info_modelo["name"], len(amostra))
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


if __name__ == "__main__":
    main()
